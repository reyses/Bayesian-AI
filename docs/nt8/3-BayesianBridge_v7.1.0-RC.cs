// =============================================================================
// _3_BayesianBridge_v71 7.1.0 -- 2026-05-20
// =============================================================================
//
// CHANGELOG 7.1.0:
//   - DATA SERIES DECOUPLED FROM HOST CHART. The bridge now streams + trades
//     a configurable TargetInstrument regardless of which chart the indicator
//     is hosted on. Drop it on ANY chart -- it always streams MNQ (or whatever
//     TargetInstrument is set to).
//   - New "Target Instrument" property (default "MNQ 06-26").
//   - Configure adds explicit data series for TargetInstrument:
//       idx 0 = host chart   (IGNORED -- never streamed)
//       idx 1 = 5s anchor    idx 2 = 1m    idx 3 = 1h
//   - Bar JSON, order routing (CreateOrder), FindPosition, and instrument
//     safety checks all use the resolved target instrument, not the chart.
//   - OnPositionUpdate filters to the target instrument + prices unrealized
//     PnL off the 5s series close (not the host chart's Close[0]).
//   - Safe-fail: if TargetInstrument cannot be resolved, DataLoaded prints
//     FATAL and the server never starts -- no silent wrong-instrument trading.
//
// CHANGELOG 7.0.0 (2026-04-15):
//   - PLACE_ORDER reads "position_effect" field (OPEN | REDUCE)
//   - REDUCE orders REJECTED if no matching opposing position exists
//   - Fixes chain-exit bug (CHEXIT_* opening opposing positions)
//
// =============================================================================
// _3_BayesianBridge_v71 — NinjaTrader 8 NinjaScript Indicator
//
// PURPOSE: TCP server inside NT8 that bridges to the Python live trading engine.
//   - Streams completed bars for TargetInstrument (5s/1m/1h) to Python
//   - Streams PARTIAL_BAR for forming bars (higher TFs on child close,
//     sub-minute on throttled ticks) so TBN workers get fresh beliefs
//   - Receives PLACE_ORDER / CLOSE_POSITION / CANCEL_ORDER from Python
//   - Sends FILL / ORDER_STATUS / POSITION messages back to Python
//
// INSTALLATION:
//   1. Copy this file to: Documents\NinjaTrader 8\bin\Custom\Indicators\
//   2. In NT8: Tools > NinjaScript Editor > right-click > Compile
//   3. Add indicator to ANY chart (instrument/TF no longer matters)
//   4. Set the "Target Instrument" property to the MNQ front month
//   5. Start the Python live engine: python -m live.engine_v2 --engine-mode l5
//
// PROTOCOL: Length-prefixed JSON over TCP (port 5199)
//   Wire format: [4 bytes: uint32 big-endian payload length][N bytes: UTF-8 JSON]
//
// IMPORTANT: This is a REFERENCE implementation. Test thoroughly on sim
// before any live deployment.
// =============================================================================

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class _3_BayesianBridge_v71 : Indicator
    {
        // ── Settings ──────────────────────────────────────────────────
        [NinjaScriptProperty]
        [Display(Name = "Port", Description = "TCP server port", Order = 1, GroupName = "Bridge")]
        public int Port { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Account Name", Description = "Trading account (e.g. DEMO6872628)", Order = 2, GroupName = "Bridge")]
        public string AccountName { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "DOM Levels", Description = "Depth of Market levels to track (0 = disabled)", Order = 3, GroupName = "Bridge")]
        public int DomLevels { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Target Instrument", Description = "Instrument to stream + trade, INDEPENDENT of the host chart (e.g. MNQ 06-26)", Order = 4, GroupName = "Bridge")]
        public string TargetInstrument { get; set; }

        // ── Version ──────────────────────────────────────────────────
        // 7.1.0: data series decoupled from host chart. The bridge now
        //        streams + trades TargetInstrument regardless of which
        //        chart the indicator is hosted on. Host chart series
        //        (BarsInProgress index 0) is ignored entirely.
        private const string BRIDGE_VERSION = "7.1.0";

        // ── Internal State ────────────────────────────────────────────
        private TcpListener  _listener;
        private TcpClient    _client;
        private NetworkStream _stream;
        private Thread        _serverThread;
        private Thread        _readThread;
        private volatile bool _running;
        private volatile bool _clientAlive;   // set false by ReadLoop on disconnect
        private Account       _account;
        private readonly object _sendLock = new object();
        private readonly Queue<Dictionary<string, string>> _inboundQueue
            = new Queue<Dictionary<string, string>>();

        // Map BarsInProgress index -> period label for the BAR message.
        // v7.1.0: index 0 = HOST CHART (ignored). Indices 1+ = the
        // explicitly-added TargetInstrument data series.
        private string[] _barLabels;
        private int[]    _barPeriodSecs;
        private int      _primaryPeriodSecs;  // host chart period (informational only)

        // The instrument the bridge actually streams + trades, resolved
        // from the first added data series (BarsArray[ANCHOR_IDX]).
        // Decoupled from the host chart's Instrument.
        private Instrument _targetInstr;
        private const int  ANCHOR_IDX = 1;   // BarsInProgress index of the 5s anchor series

        // History buffer — completed bars (TFs 1-11 only, skip 1s) stored
        // here; dumped to client on connect so Python bypasses warmup.
        // Capped at MAX_HISTORY to avoid memory blowup with 1-year charts.
        private readonly List<string> _allBars = new List<string>();
        private readonly object _barLock = new object();
        private const int MAX_HISTORY = 10000;   // ~2MB, 1h buffer for reconnects (Python persists full history)

        // DMI: Value = (DI+ - DI-) / (DI+ + DI-), range [-1, +1]
        // DM: DiPlus = DI+ (0-100), DiMinus = DI- (0-100), ADXPlot = ADX
        // ADX: Average Directional Index, range [0, 100]. <20 = chop, >25 = trend.
        private const int DMI_PERIOD = 14;
        private const int SE_PERIOD = 60;
        private NinjaTrader.NinjaScript.Indicators.DMI[] _dmiInd;
        private NinjaTrader.NinjaScript.Indicators.DM[] _dmInd;
        private NinjaTrader.NinjaScript.Indicators.ADX[] _adxInd;
        private NinjaTrader.NinjaScript.Indicators.StdError[] _seInd;

        // DOM throttle — send at most every N ms to avoid flooding
        private DateTime _lastDomSend = DateTime.MinValue;
        private const int DOM_THROTTLE_MS = 250;

        // Account equity update throttle
        private DateTime _lastAccountSend = DateTime.MinValue;
        private const int ACCOUNT_UPDATE_MS = 5000;  // every 5 seconds

        // Partial bar throttle for sub-minute tick feed
        private DateTime _lastPartialSend = DateTime.MinValue;
        private const int PARTIAL_THROTTLE_MS = 250;

        // Latest best bid/ask (updated on every depth event, sent throttled)
        private double _bestBid, _bestAsk;
        private long   _bestBidSize, _bestAskSize;

        // ── NinjaScript Lifecycle ─────────────────────────────────────

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "TCP bridge to Bayesian-AI Python live engine";
                Name        = "3-BayesianBridge_v7.1.0";
                IsOverlay   = true;
                Port        = 5199;
                AccountName = "Sim101";
                DomLevels   = 5;
                TargetInstrument = "MNQ 06-26";
                MaximumBarsLookBack = MaximumBarsLookBack.Infinite;
                Calculate           = Calculate.OnBarClose;
            }
            else if (State == State.Configure)
            {
                // ── v7.1.0: explicit TargetInstrument data series ───────
                // The host chart (BarsInProgress index 0) is IGNORED. We
                // add our own series for TargetInstrument so the bridge
                // streams + trades the same instrument no matter which
                // chart the indicator is dropped on.
                //
                //   idx 0 = host chart            (ignored)
                //   idx 1 = TargetInstrument 5s   (ANCHOR_IDX)
                //   idx 2 = TargetInstrument 1m
                //   idx 3 = TargetInstrument 1h
                _primaryPeriodSecs = GetBarsPeriodSeconds(BarsPeriod);  // informational

                AddDataSeries(TargetInstrument, BarsPeriodType.Second, 5);
                AddDataSeries(TargetInstrument, BarsPeriodType.Minute, 1);
                AddDataSeries(TargetInstrument, BarsPeriodType.Minute, 60);

                // Index-aligned with BarsInProgress. Slot 0 is the host
                // chart placeholder -- never emitted (OnBarUpdate skips it).
                _barLabels     = new string[] { "(host)", "5s", "1m", "1h" };
                _barPeriodSecs = new int[]    { 0,        5,    60,   3600 };
            }
            else if (State == State.DataLoaded)
            {
                // Resolve the target instrument from the anchor data series.
                // BarsArray[ANCHOR_IDX] is the 5s TargetInstrument series
                // added in Configure -- its Instrument is what we trade.
                if (BarsArray != null && BarsArray.Length > ANCHOR_IDX
                    && BarsArray[ANCHOR_IDX] != null)
                {
                    _targetInstr = BarsArray[ANCHOR_IDX].Instrument;
                }
                if (_targetInstr == null)
                {
                    Print("_3_BayesianBridge_v71: FATAL — could not resolve TargetInstrument '"
                        + TargetInstrument + "'. Check the contract name.");
                    return;
                }

                // Find the account
                lock (Account.All)
                {
                    foreach (Account acct in Account.All)
                    {
                        if (acct.Name == AccountName)
                        {
                            _account = acct;
                            break;
                        }
                    }
                }

                if (_account == null)
                {
                    Print("_3_BayesianBridge_v71: Account '" + AccountName + "' not found!");
                    return;
                }

                // Initialize DMI + DM + ADX + StdError indicators for each data series
                _dmiInd = new NinjaTrader.NinjaScript.Indicators.DMI[_barLabels.Length];
                _dmInd = new NinjaTrader.NinjaScript.Indicators.DM[_barLabels.Length];
                _adxInd = new NinjaTrader.NinjaScript.Indicators.ADX[_barLabels.Length];
                _seInd = new NinjaTrader.NinjaScript.Indicators.StdError[_barLabels.Length];
                for (int i = 0; i < _barLabels.Length; i++)
                {
                    _dmiInd[i] = DMI(Closes[i], DMI_PERIOD);
                    _dmInd[i] = DM(BarsArray[i], DMI_PERIOD);
                    _adxInd[i] = ADX(BarsArray[i], DMI_PERIOD);
                    _seInd[i] = StdError(Closes[i], SE_PERIOD);
                }

                // Subscribe to execution events
                _account.OrderUpdate     += OnOrderUpdate;
                _account.ExecutionUpdate += OnExecutionUpdate;
                _account.PositionUpdate  += OnPositionUpdate;

                // Start TCP server
                _running = true;
                _serverThread = new Thread(ServerLoop) { IsBackground = true };
                _serverThread.Start();

                Print("_3_BayesianBridge_v71 " + BRIDGE_VERSION + ": Started on port " + Port
                    + ", account=" + AccountName
                    + ", target=" + _targetInstr.FullName
                    + ", host_chart=" + GetTFLabel(_primaryPeriodSecs)
                    + " (streaming " + (_barLabels.Length - 1) + " series: "
                    + string.Join(",", _barLabels, 1, _barLabels.Length - 1) + ")");
            }
            else if (State == State.Terminated)
            {
                _running = false;

                if (_account != null)
                {
                    _account.OrderUpdate     -= OnOrderUpdate;
                    _account.ExecutionUpdate -= OnExecutionUpdate;
                    _account.PositionUpdate  -= OnPositionUpdate;
                }

                try { _client?.Close(); }   catch { }
                try { _listener?.Stop(); }  catch { }

                Print("_3_BayesianBridge_v71: Stopped");
            }
        }

        // ── Connection Status ─────────────────────────────────────────

        private bool _connectionLost = false;

        protected override void OnConnectionStatusUpdate(ConnectionStatusEventArgs e)
        {
            if (e.Status == ConnectionStatus.ConnectionLost)
            {
                _connectionLost = true;
                Print("_3_BayesianBridge_v71: CONNECTION LOST — notifying Python");
                SendRawJson("{" + Q("type") + ":" + Q("CONNECTION_LOST") + "}");
            }
            else if (e.Status == ConnectionStatus.Disconnected)
            {
                _connectionLost = true;
                Print("_3_BayesianBridge_v71: DISCONNECTED — notifying Python");
                SendRawJson("{" + Q("type") + ":" + Q("CONNECTION_LOST") + "}");
            }
            else if (e.Status == ConnectionStatus.Connected)
            {
                if (_connectionLost)
                {
                    _connectionLost = false;
                    Print("_3_BayesianBridge_v71: CONNECTION RESTORED — notifying Python");
                    SendRawJson("{" + Q("type") + ":" + Q("CONNECTION_RESTORED") + "}");
                }
            }
        }

        // ── Bar Updates ───────────────────────────────────────────────

        private DateTime _lastBarTime = DateTime.MinValue;
        private bool _haltNotified = false;

        protected override void OnBarUpdate()
        {
            int idx = BarsInProgress;
            if (_barLabels == null || idx >= _barLabels.Length)
                return;
            // v7.1.0: index 0 is the host chart -- never streamed. Only the
            // explicitly-added TargetInstrument series (idx 1+) are emitted.
            if (idx == 0 || _targetInstr == null)
                return;
            if (CurrentBars[idx] < 1 || IsFirstTickOfBar == false)
                return;
            // Block data during Playback or connection loss
            if (State == State.Historical || _connectionLost)
                return;

            // Market halt detection: if >2 min gap since last bar, notify
            DateTime now = DateTime.Now;
            if (_lastBarTime != DateTime.MinValue)
            {
                double gap = (now - _lastBarTime).TotalSeconds;
                if (gap > 15 && !_haltNotified)
                {
                    Print("_3_BayesianBridge_v71: MARKET HALT detected (" + gap.ToString("F0") + "s gap)");
                    SendRawJson("{" + Q("type") + ":" + Q("MARKET_HALT") + ","
                        + Q("gap_seconds") + ":" + gap.ToString("F0") + "}");
                    _haltNotified = true;
                }
                else if (gap <= 15 && _haltNotified)
                {
                    Print("_3_BayesianBridge_v71: MARKET RESUMED (ticks flowing)");
                    SendRawJson("{" + Q("type") + ":" + Q("MARKET_RESUMED") + "}");
                    _haltNotified = false;
                }
            }
            _lastBarTime = now;

            // Build bar JSON for whichever TF just completed
            // Includes DMI+/DMI-/ADX from NT8 native indicators
            // DMI + ADX from NT8 native indicators
            double dmiVal = 0, adxVal = 0, dmiPlus = 0, dmiMinus = 0;
            if (_dmiInd != null && idx < _dmiInd.Length && _dmiInd[idx] != null
                && CurrentBars[idx] >= DMI_PERIOD)
            {
                try { dmiVal = _dmiInd[idx].Value[1]; }
                catch { }
            }
            if (_dmInd != null && idx < _dmInd.Length && _dmInd[idx] != null
                && CurrentBars[idx] >= DMI_PERIOD)
            {
                try { dmiPlus = _dmInd[idx].DiPlus[1]; }
                catch { }
                try { dmiMinus = _dmInd[idx].DiMinus[1]; }
                catch { }
            }
            if (_adxInd != null && idx < _adxInd.Length && _adxInd[idx] != null
                && CurrentBars[idx] >= DMI_PERIOD)
            {
                try { adxVal = _adxInd[idx].Value[1]; }
                catch { }
            }

            // StdError: regression line + upper/lower bands (observational, for parity logging)
            double seMiddle = 0, seUpper = 0, seLower = 0;
            if (_seInd != null && idx < _seInd.Length && _seInd[idx] != null
                && CurrentBars[idx] >= SE_PERIOD)
            {
                try { seMiddle = _seInd[idx].Middle[1]; }
                catch { }
                try { seUpper = _seInd[idx].Upper[1]; }
                catch { }
                try { seLower = _seInd[idx].Lower[1]; }
                catch { }
            }

            // Base JSON without live flag (stored in history buffer)
            string jsonBase = "{"
                + Q("type") + ":" + Q("BAR") + ","
                + Q("instrument") + ":" + Q(_targetInstr.FullName) + ","
                + Q("tf") + ":" + Q(_barLabels[idx]) + ","
                + Q("bar_period_s") + ":" + _barPeriodSecs[idx] + ","
                + Q("timestamp") + ":" + D2S(ToUnixSeconds(Times[idx][1])) + ","
                + Q("open") + ":" + D2S(Opens[idx][1]) + ","
                + Q("high") + ":" + D2S(Highs[idx][1]) + ","
                + Q("low") + ":" + D2S(Lows[idx][1]) + ","
                + Q("close") + ":" + D2S(Closes[idx][1]) + ","
                + Q("volume") + ":" + D2S(Volumes[idx][1]) + ","
                + Q("dmi") + ":" + D2S(dmiVal) + ","
                + Q("dmi_plus") + ":" + D2S(dmiPlus) + ","
                + Q("dmi_minus") + ":" + D2S(dmiMinus) + ","
                + Q("adx") + ":" + D2S(adxVal) + ","
                + Q("se_mid") + ":" + D2S(seMiddle) + ","
                + Q("se_upper") + ":" + D2S(seUpper) + ","
                + Q("se_lower") + ":" + D2S(seLower)
                + "}";

            // Buffer bars >= 5s for history dump (WITHOUT live flag)
            if (_barPeriodSecs[idx] >= 5)
            {
                lock (_barLock)
                {
                    _allBars.Add(jsonBase);
                    if (_allBars.Count > MAX_HISTORY)
                        _allBars.RemoveRange(0, _allBars.Count - MAX_HISTORY);
                }
            }
            // Send WITH live flag to Python
            string jsonLive = jsonBase.Substring(0, jsonBase.Length - 1)
                + "," + Q("live") + ":" + "true" + "}";
            SendRawJson(jsonLive);

            // Send PARTIAL_BAR for all higher TF series (their forming bar)
            for (int hi = idx + 1; hi < _barPeriodSecs.Length; hi++)
            {
                if (CurrentBars[hi] < 1) continue;
                double hiDmi = 0, hiAdx = 0, hiDmiP = 0, hiDmiM = 0;
                if (_dmiInd != null && hi < _dmiInd.Length && _dmiInd[hi] != null
                    && CurrentBars[hi] >= DMI_PERIOD)
                {
                    try { hiDmi = _dmiInd[hi].Value[0]; }
                    catch { }
                }
                if (_dmInd != null && hi < _dmInd.Length && _dmInd[hi] != null
                    && CurrentBars[hi] >= DMI_PERIOD)
                {
                    try { hiDmiP = _dmInd[hi].DiPlus[0]; }
                    catch { }
                    try { hiDmiM = _dmInd[hi].DiMinus[0]; }
                    catch { }
                }
                if (_adxInd != null && hi < _adxInd.Length && _adxInd[hi] != null
                    && CurrentBars[hi] >= DMI_PERIOD)
                {
                    try { hiAdx = _adxInd[hi].Value[0]; }
                    catch { }
                }
                string partial = "{"
                    + Q("type") + ":" + Q("PARTIAL_BAR") + ","
                    + Q("instrument") + ":" + Q(_targetInstr.FullName) + ","
                    + Q("tf") + ":" + Q(_barLabels[hi]) + ","
                    + Q("bar_period_s") + ":" + _barPeriodSecs[hi] + ","
                    + Q("timestamp") + ":" + D2S(ToUnixSeconds(Times[hi][0])) + ","
                    + Q("open") + ":" + D2S(Opens[hi][0]) + ","
                    + Q("high") + ":" + D2S(Highs[hi][0]) + ","
                    + Q("low") + ":" + D2S(Lows[hi][0]) + ","
                    + Q("close") + ":" + D2S(Closes[hi][0]) + ","
                    + Q("volume") + ":" + D2S(Volumes[hi][0]) + ","
                    + Q("dmi") + ":" + D2S(hiDmi) + ","
                    + Q("dmi_plus") + ":" + D2S(hiDmiP) + ","
                    + Q("dmi_minus") + ":" + D2S(hiDmiM) + ","
                    + Q("adx") + ":" + D2S(hiAdx)
                    + "}";
                SendRawJson(partial);
            }
        }

        // ── Partial bars for sub-minute TFs on each trade tick ──────

        protected override void OnMarketData(MarketDataEventArgs e)
        {
            if (e.MarketDataType != MarketDataType.Last) return;
            if (_client == null || !_clientAlive) return;

            DateTime now = DateTime.UtcNow;
            if ((now - _lastPartialSend).TotalMilliseconds < PARTIAL_THROTTLE_MS)
                return;
            _lastPartialSend = now;

            // Start at i=1 -- index 0 is the host chart placeholder (period 0).
            for (int i = 1; i < _barPeriodSecs.Length; i++)
            {
                if (_barPeriodSecs[i] >= 60) break;  // sorted ascending — stop at 1m+
                if (CurrentBars[i] < 1) continue;
                string partial = "{"
                    + Q("type") + ":" + Q("PARTIAL_BAR") + ","
                    + Q("instrument") + ":" + Q(_targetInstr.FullName) + ","
                    + Q("tf") + ":" + Q(_barLabels[i]) + ","
                    + Q("bar_period_s") + ":" + _barPeriodSecs[i] + ","
                    + Q("timestamp") + ":" + D2S(ToUnixSeconds(Times[i][0])) + ","
                    + Q("open") + ":" + D2S(Opens[i][0]) + ","
                    + Q("high") + ":" + D2S(Highs[i][0]) + ","
                    + Q("low") + ":" + D2S(Lows[i][0]) + ","
                    + Q("close") + ":" + D2S(Closes[i][0]) + ","
                    + Q("volume") + ":" + D2S(Volumes[i][0])
                    + "}";
                SendRawJson(partial);
            }
        }

        // ── DOM (Depth of Market) ────────────────────────────────────

        protected override void OnMarketDepth(MarketDepthEventArgs e)
        {
            if (DomLevels <= 0) return;   // DOM disabled

            // Track best bid/ask from top-of-book updates
            if (e.Position == 0)
            {
                if (e.MarketDataType == MarketDataType.Bid)
                {
                    _bestBid     = e.Price;
                    _bestBidSize = e.Volume;
                }
                else if (e.MarketDataType == MarketDataType.Ask)
                {
                    _bestAsk     = e.Price;
                    _bestAskSize = e.Volume;
                }
            }

            // Throttle: send snapshot at most every DOM_THROTTLE_MS
            if ((DateTime.Now - _lastDomSend).TotalMilliseconds < DOM_THROTTLE_MS)
                return;
            _lastDomSend = DateTime.Now;

            if (_bestBid <= 0 || _bestAsk <= 0) return;

            double imbalance = (_bestBidSize + _bestAskSize) > 0
                ? (double)(_bestBidSize - _bestAskSize) / (_bestBidSize + _bestAskSize)
                : 0.0;

            string json = "{"
                + Q("type") + ":" + Q("DOM") + ","
                + Q("instrument") + ":" + Q(_targetInstr.FullName) + ","
                + Q("bid") + ":" + D2S(_bestBid) + ","
                + Q("bid_size") + ":" + _bestBidSize + ","
                + Q("ask") + ":" + D2S(_bestAsk) + ","
                + Q("ask_size") + ":" + _bestAskSize + ","
                + Q("spread") + ":" + D2S(_bestAsk - _bestBid) + ","
                + Q("imbalance") + ":" + D2S(imbalance) + ","
                + Q("timestamp") + ":" + D2S(ToUnixSeconds(DateTime.UtcNow))
                + "}";
            SendRawJson(json);
        }

        // ── Account Event Handlers ────────────────────────────────────

        private void OnOrderUpdate(object sender, OrderEventArgs e)
        {
            string status = e.Order.OrderState.ToString();
            string oid = e.Order.Name ?? "";
            string json = "{"
                + Q("type") + ":" + Q("ORDER_STATUS") + ","
                + Q("order_id") + ":" + Q(oid) + ","
                + Q("status") + ":" + Q(status)
                + "}";
            SendRawJson(json);

            // Log all order state changes for our orders
            if (oid.StartsWith("ENTRY") || oid.StartsWith("CHAIN") || oid.StartsWith("CHEXIT")
                || oid == "BAY_CLOSE" || oid.StartsWith("BAY_"))
            {
                Print("   ORDER " + oid + " -> " + status);
            }
        }

        // Session tracking for NT8 output
        private int _sessionTrades = 0;
        private int _sessionWins = 0;
        private double _sessionPnl = 0;
        private double _lastEntryPrice = 0;
        private string _lastEntrySide = "";

        private void OnExecutionUpdate(object sender, ExecutionEventArgs e)
        {
            if (e.Execution == null) return;

            string side = e.Execution.MarketPosition == MarketPosition.Long ? "BUY" : "SELL";
            string json = "{"
                + Q("type") + ":" + Q("FILL") + ","
                + Q("order_id") + ":" + Q(e.Execution.Order.Name) + ","
                + Q("side") + ":" + Q(side) + ","
                + Q("qty") + ":" + e.Execution.Quantity + ","
                + Q("fill_price") + ":" + D2S(e.Execution.Price) + ","
                + Q("fill_time") + ":" + D2S(ToUnixSeconds(e.Execution.Time)) + ","
                + Q("commission") + ":" + D2S(e.Execution.Commission)
                + "}";
            SendRawJson(json);

            // Verbose trade logging to NT8 output
            string orderName = e.Execution.Order.Name ?? "";
            bool isEntry = orderName.StartsWith("ENTRY_") || orderName.StartsWith("BAY_");
            bool isChainEntry = orderName.StartsWith("CHAIN_");
            bool isChainExit = orderName.StartsWith("CHEXIT_");
            bool isClose = orderName == "BAY_CLOSE" || orderName == "Close";

            if (isEntry)
            {
                _lastEntryPrice = e.Execution.Price;
                _lastEntrySide = side;
                Print(">> ENTRY " + side + " @ " + e.Execution.Price.ToString("F2")
                    + "  id=" + orderName
                    + "  [" + e.Execution.Time.ToString("HH:mm:ss") + "]");
            }
            else if (isChainEntry)
            {
                Print(">> CHAIN " + side + " @ " + e.Execution.Price.ToString("F2")
                    + "  id=" + orderName
                    + "  [" + e.Execution.Time.ToString("HH:mm:ss") + "]");
            }
            else if (isChainExit && _lastEntryPrice > 0)
            {
                // Chain exit = partial close. Compute PnL for 1 contract, emit event.
                double cpnl = 0;
                if (_lastEntrySide == "BUY")
                    cpnl = (e.Execution.Price - _lastEntryPrice) * 2.0;
                else
                    cpnl = (_lastEntryPrice - e.Execution.Price) * 2.0;

                Print("<< CHAIN EXIT @ " + e.Execution.Price.ToString("F2")
                    + "  PnL=" + (cpnl >= 0 ? "+$" : "-$") + Math.Abs(cpnl).ToString("F2")
                    + "  id=" + orderName
                    + "  [" + e.Execution.Time.ToString("HH:mm:ss") + "]");

                SendTradeClosed(orderName, _lastEntrySide,
                    _lastEntryPrice, e.Execution.Price, cpnl,
                    e.Execution.Time, 1, true);
            }
            else if (isClose && _lastEntryPrice > 0)
            {
                double pnl = 0;
                if (_lastEntrySide == "BUY")
                    pnl = (e.Execution.Price - _lastEntryPrice) * 2.0;
                else
                    pnl = (_lastEntryPrice - e.Execution.Price) * 2.0;

                _sessionTrades++;
                _sessionPnl += pnl;
                if (pnl > 0) _sessionWins++;
                double wr = _sessionTrades > 0 ? (double)_sessionWins / _sessionTrades * 100 : 0;

                string pnlStr = pnl >= 0 ? "+$" + pnl.ToString("F2") : "-$" + Math.Abs(pnl).ToString("F2");
                Print("<< EXIT  " + _lastEntrySide + " @ " + e.Execution.Price.ToString("F2")
                    + "  PnL=" + pnlStr
                    + "  [" + e.Execution.Time.ToString("HH:mm:ss") + "]");
                Print("   Session: " + _sessionTrades + " trades  WR=" + wr.ToString("F0") + "%"
                    + "  PnL=$" + _sessionPnl.ToString("F2"));

                SendTradeClosed(orderName, _lastEntrySide,
                    _lastEntryPrice, e.Execution.Price, pnl,
                    e.Execution.Time, e.Execution.Quantity, false);

                _lastEntryPrice = 0;
            }
            else
            {
                // Unknown order — log everything for debug
                Print("?? FILL " + side + " x" + e.Execution.Quantity
                    + " @ " + e.Execution.Price.ToString("F2")
                    + "  id=" + orderName
                    + "  [" + e.Execution.Time.ToString("HH:mm:ss") + "]");
            }
        }

        private void OnPositionUpdate(object sender, PositionEventArgs e)
        {
            if (e.Position == null) return;

            // v7.1.0: ignore position updates for instruments we don't trade.
            // The account may hold positions in other symbols; the bridge
            // only reports/manages TargetInstrument.
            if (_targetInstr != null
                && e.Position.Instrument.FullName != _targetInstr.FullName)
                return;

            try
            {
                int qty = 0;
                if (e.Position.MarketPosition == MarketPosition.Long)
                    qty = e.Position.Quantity;
                else if (e.Position.MarketPosition == MarketPosition.Short)
                    qty = -e.Position.Quantity;

                // Unrealized PnL priced off the target series' last close
                // (NOT the host chart's Close[0]).
                double refPx = 0;
                try { refPx = Closes[ANCHOR_IDX][0]; }
                catch { }
                double unrealPnl = 0;
                try { unrealPnl = e.Position.GetUnrealizedProfitLoss(
                    PerformanceUnit.Currency, refPx); }
                catch { }  // ref price may not be available yet

                string json = "{"
                    + Q("type") + ":" + Q("POSITION") + ","
                    + Q("instrument") + ":" + Q(e.Position.Instrument.FullName) + ","
                    + Q("qty") + ":" + qty + ","
                    + Q("avg_price") + ":" + D2S(e.Position.AveragePrice) + ","
                    + Q("unrealized_pnl") + ":" + D2S(unrealPnl)
                    + "}";
                SendRawJson(json);

                // Also send full account equity on position changes
                SendAccountUpdate();
            }
            catch (Exception ex)
            {
                Print("_3_BayesianBridge_v71: OnPositionUpdate error: " + ex.Message);
            }
        }

        // ── TCP Server ────────────────────────────────────────────────

        private void ServerLoop()
        {
            try
            {
                _listener = new TcpListener(IPAddress.Loopback, Port);
                _listener.Start();
                Print("_3_BayesianBridge_v71: Listening on 127.0.0.1:" + Port);

                while (_running)
                {
                    if (!_listener.Pending())
                    {
                        Thread.Sleep(100);
                        continue;
                    }

                    // Close any stale previous connection before accepting
                    if (_client != null)
                    {
                        Print("_3_BayesianBridge_v71: Closing previous client connection");
                        _clientAlive = false;
                        try { _stream?.Close(); } catch { }
                        try { _client?.Close(); } catch { }
                        _stream = null;
                        _client = null;
                    }

                    _client = _listener.AcceptTcpClient();
                    _stream = _client.GetStream();
                    _clientAlive = true;
                    Print("_3_BayesianBridge_v71: Python client connected");

                    try
                    {
                        // Send CONNECTED message
                        Print("_3_BayesianBridge_v71: Sending CONNECTED...");
                        string connJson = "{"
                            + Q("type") + ":" + Q("CONNECTED") + ","
                            + Q("account") + ":" + Q(AccountName) + ","
                            + Q("instrument") + ":" + Q(_targetInstr.FullName) + ","
                            + Q("primary_period_s") + ":" + _primaryPeriodSecs + ","
                            + Q("version") + ":" + Q(BRIDGE_VERSION)
                            + "}";
                        SendRawJson(connJson);
                        Print("_3_BayesianBridge_v71: CONNECTED sent OK");

                        Print("_3_BayesianBridge_v71: Sending position snapshot...");
                        SendPositionSnapshot();
                        Print("_3_BayesianBridge_v71: Sending account update...");
                        SendAccountUpdate();
                        // History is no longer sent automatically on connect.
                        // Python sends REQUEST_HISTORY when it needs the dump.
                        Print("_3_BayesianBridge_v71: Ready (history on request)");
                    }
                    catch (Exception ex)
                    {
                        Print("_3_BayesianBridge_v71: Init send failed: " + ex.ToString());
                    }

                    _readThread = new Thread(ReadLoop) { IsBackground = true };
                    _readThread.Start();

                    // _clientAlive is set to false by ReadLoop when it
                    // detects a disconnect (read returns 0 or throws).
                    // Do NOT drain pending connections here — the outer loop
                    // handles new connections properly after _clientAlive goes false.
                    while (_running && _clientAlive)
                    {
                        try
                        {
                            ProcessInboundQueue();

                            // Periodic account equity update
                            if ((DateTime.Now - _lastAccountSend).TotalMilliseconds >= ACCOUNT_UPDATE_MS)
                            {
                                SendAccountUpdate();
                                _lastAccountSend = DateTime.Now;
                            }
                        }
                        catch (Exception ex)
                        {
                            Print("_3_BayesianBridge_v71: Loop error: " + ex.Message);
                        }

                        Thread.Sleep(50);
                    }

                    Print("_3_BayesianBridge_v71: Python client disconnected");
                }
            }
            catch (Exception ex)
            {
                if (_running)
                    Print("_3_BayesianBridge_v71: Server FATAL: " + ex.ToString());
            }
        }

        private void ReadLoop()
        {
            // Capture the stream we were started with — if the ServerLoop
            // recycles to a new client, we must NOT clobber _clientAlive.
            NetworkStream myStream = _stream;
            Print("_3_BayesianBridge_v71: ReadLoop started (stream=" + myStream?.GetHashCode() + ")");
            try
            {
                while (_running && _clientAlive && myStream != null)
                {
                    byte[] header = ReadExactly(myStream, 4);
                    if (header == null) break;

                    int length = (header[0] << 24) | (header[1] << 16)
                               | (header[2] << 8)  | header[3];

                    if (length <= 0 || length > 1048576) break;

                    byte[] payload = ReadExactly(myStream, length);
                    if (payload == null) break;

                    string json = Encoding.UTF8.GetString(payload);
                    var msg = ParseSimpleJson(json);

                    lock (_inboundQueue)
                    {
                        _inboundQueue.Enqueue(msg);
                    }
                }
            }
            catch (Exception ex)
            {
                if (_running)
                    Print("_3_BayesianBridge_v71: Read error: " + ex.Message);
            }

            // Only signal dead if WE are still the active connection.
            // If ServerLoop already recycled to a new client, don't clobber.
            if (_stream == myStream)
                _clientAlive = false;
        }

        private byte[] ReadExactly(NetworkStream stream, int count)
        {
            byte[] buffer = new byte[count];
            int offset = 0;
            while (offset < count)
            {
                int read = stream.Read(buffer, offset, count - offset);
                if (read == 0) return null;
                offset += read;
            }
            return buffer;
        }

        // ── Inbound Command Processing ────────────────────────────────

        private void ProcessInboundQueue()
        {
            List<Dictionary<string, string>> commands;
            lock (_inboundQueue)
            {
                if (_inboundQueue.Count == 0) return;
                commands = new List<Dictionary<string, string>>(_inboundQueue);
                _inboundQueue.Clear();
            }

            foreach (var cmd in commands)
            {
                string msgType = "";
                cmd.TryGetValue("type", out msgType);
                if (msgType == null) msgType = "";

                switch (msgType)
                {
                    case "PLACE_ORDER":
                        HandlePlaceOrder(cmd);
                        break;
                    case "CLOSE_POSITION":
                        HandleClosePosition(cmd);
                        break;
                    case "CANCEL_ORDER":
                        HandleCancelOrder(cmd);
                        break;
                    case "SUBSCRIBE":
                        string inst = "", bps = "";
                        cmd.TryGetValue("instrument", out inst);
                        cmd.TryGetValue("bar_period_s", out bps);
                        Print("_3_BayesianBridge_v71: SUBSCRIBE — instrument="
                            + inst + ", bar_period_s=" + bps);
                        break;
                    case "HEARTBEAT":
                        // Enhanced heartbeat — includes position state for drift detection
                        Position hbPos = FindPosition();
                        int hbQty = 0;
                        string hbSide = "FLAT";
                        double hbAvg = 0;
                        if (hbPos != null && hbPos.MarketPosition != MarketPosition.Flat)
                        {
                            hbQty = hbPos.Quantity;
                            hbSide = hbPos.MarketPosition == MarketPosition.Long ? "LONG" : "SHORT";
                            hbAvg = hbPos.AveragePrice;
                        }
                        string hb = "{"
                            + Q("type") + ":" + Q("HEARTBEAT") + ","
                            + Q("server_time") + ":" + D2S(ToUnixSeconds(DateTime.UtcNow)) + ","
                            + Q("position_qty") + ":" + hbQty + ","
                            + Q("position_side") + ":" + Q(hbSide) + ","
                            + Q("position_avg_price") + ":" + D2S(hbAvg)
                            + "}";
                        SendRawJson(hb);
                        break;
                    case "REQUEST_HISTORY":
                        Print("_3_BayesianBridge_v71: Sending history buffer on request...");
                        SendHistoryBuffer();
                        break;
                    case "RESUME_FROM":
                        string lastTs = GetVal(cmd, "last_timestamp", "0");
                        Print("_3_BayesianBridge_v71: Delta sync from ts=" + lastTs);
                        SendHistoryBufferFrom(double.Parse(lastTs));
                        break;
                    case "REQUEST_POSITION":
                        Print("_3_BayesianBridge_v71: Position requested by Python");
                        SendPositionSnapshot();
                        SendAccountUpdate();
                        break;
                    default:
                        Print("_3_BayesianBridge_v71: Unknown command: " + msgType);
                        break;
                }
            }
        }

        private void HandlePlaceOrder(Dictionary<string, string> cmd)
        {
            string orderId   = GetVal(cmd, "order_id", "BAY_UNK");
            string side      = GetVal(cmd, "side", "BUY");
            int    qty       = GetIntVal(cmd, "qty", 1);
            string reqInst   = GetVal(cmd, "instrument", "");
            string posEffect = GetVal(cmd, "position_effect", "OPEN");  // OPEN | REDUCE

            // Instrument safety check — reject if Python wants a different symbol
            if (reqInst.Length > 0
                && !_targetInstr.FullName.ToUpper().Contains(reqInst.Split(' ')[0].ToUpper()))
            {
                Print("_3_BayesianBridge_v71: REJECTED PLACE_ORDER — instrument mismatch: "
                    + "requested=" + reqInst + " target=" + _targetInstr.FullName);
                SendOrderRejected(orderId, "instrument_mismatch");
                return;
            }

            Print("_3_BayesianBridge_v71: PLACE_ORDER " + side + " " + qty
                + " effect=" + posEffect + " id=" + orderId);

            // ACK — confirm receipt before submitting
            SendOrderAck(orderId);

            if (_account == null)
            {
                Print("_3_BayesianBridge_v71: REJECTED — _account is null (check AccountName setting)");
                SendOrderRejected(orderId, "account_null");
                return;
            }

            // ── Pick OrderAction based on side + position_effect ──────────
            // OPEN:  Buy / SellShort       (open or add to position)
            // REDUCE: Sell / BuyToCover    (close existing opposing position)
            //
            // REDUCE is REJECTED if there is no matching opposing position —
            // we never want to silently turn a "close my long" into "open a
            // new short" (the bug that motivated 7.0.0).
            OrderAction action;
            if (posEffect == "REDUCE")
            {
                Position pos = FindPosition();
                bool isLong  = pos != null && pos.MarketPosition == MarketPosition.Long
                               && pos.Quantity > 0;
                bool isShort = pos != null && pos.MarketPosition == MarketPosition.Short
                               && pos.Quantity > 0;

                if (side == "SELL" && isLong)
                {
                    action = OrderAction.Sell;
                }
                else if (side == "BUY" && isShort)
                {
                    action = OrderAction.BuyToCover;
                }
                else
                {
                    string posStr = pos == null || pos.MarketPosition == MarketPosition.Flat
                        ? "FLAT"
                        : (pos.MarketPosition + " x" + pos.Quantity);
                    Print("_3_BayesianBridge_v71: REJECTED REDUCE — side=" + side
                        + " position=" + posStr + " (no matching opposing position)");
                    SendOrderRejected(orderId, "reduce_no_matching_position");
                    return;
                }

                // Cap qty at current position size — never accidentally over-close
                if (qty > pos.Quantity)
                {
                    Print("_3_BayesianBridge_v71: REDUCE qty=" + qty + " > position=" + pos.Quantity
                        + " — capping to position size");
                    qty = pos.Quantity;
                }
            }
            else  // OPEN (default)
            {
                action = side == "BUY" ? OrderAction.Buy : OrderAction.SellShort;
            }

            try
            {
                Order order = _account.CreateOrder(
                    _targetInstr,
                    action,
                    OrderType.Market,
                    OrderEntry.Manual,
                    TimeInForce.Gtc,
                    qty,
                    0, 0,
                    "",
                    orderId,
                    DateTime.MaxValue,
                    null
                );
                _account.Submit(new[] { order });
            }
            catch (Exception ex)
            {
                Print("_3_BayesianBridge_v71: Order submit failed: " + ex.Message);
                SendOrderRejected(orderId, ex.Message);
            }
        }

        private void HandleClosePosition(Dictionary<string, string> cmd)
        {
            string reqInst = GetVal(cmd, "instrument", "");

            // Instrument safety check
            if (reqInst.Length > 0
                && !_targetInstr.FullName.ToUpper().Contains(reqInst.Split(' ')[0].ToUpper()))
            {
                Print("_3_BayesianBridge_v71: REJECTED CLOSE_POSITION — instrument mismatch: "
                    + "requested=" + reqInst + " target=" + _targetInstr.FullName);
                return;
            }

            Print("_3_BayesianBridge_v71: CLOSE_POSITION");

            try
            {
                Position pos = FindPosition();
                if (pos != null && pos.MarketPosition != MarketPosition.Flat)
                {
                    OrderAction action = pos.MarketPosition == MarketPosition.Long
                        ? OrderAction.Sell : OrderAction.BuyToCover;

                    Order closeOrder = _account.CreateOrder(
                        _targetInstr,
                        action,
                        OrderType.Market,
                        OrderEntry.Manual,
                        TimeInForce.Gtc,
                        pos.Quantity,
                        0, 0,
                        "",
                        "BAY_CLOSE",
                        DateTime.MaxValue,
                        null
                    );
                    _account.Submit(new[] { closeOrder });
                }
                else
                {
                    Print("_3_BayesianBridge_v71: No position to close");
                }
            }
            catch (Exception ex)
            {
                Print("_3_BayesianBridge_v71: Close position failed: " + ex.Message);
            }
        }

        private void HandleCancelOrder(Dictionary<string, string> cmd)
        {
            string orderId = GetVal(cmd, "order_id", "");
            Print("_3_BayesianBridge_v71: CANCEL_ORDER " + orderId);

            try
            {
                foreach (Order order in _account.Orders)
                {
                    if (order.Name == orderId
                        && (order.OrderState == OrderState.Accepted
                            || order.OrderState == OrderState.Working))
                    {
                        _account.Cancel(new[] { order });
                        return;
                    }
                }
                Print("_3_BayesianBridge_v71: Order " + orderId + " not found or not cancellable");
            }
            catch (Exception ex)
            {
                Print("_3_BayesianBridge_v71: Cancel failed: " + ex.Message);
            }
        }

        // ── Outbound Messaging ────────────────────────────────────────

        private void SendRawJson(string json)
        {
            if (_stream == null || _client == null || !_client.Connected)
                return;

            try
            {
                byte[] payload = Encoding.UTF8.GetBytes(json);
                byte[] header  = new byte[4];
                header[0] = (byte)((payload.Length >> 24) & 0xFF);
                header[1] = (byte)((payload.Length >> 16) & 0xFF);
                header[2] = (byte)((payload.Length >>  8) & 0xFF);
                header[3] = (byte)((payload.Length      ) & 0xFF);

                lock (_sendLock)
                {
                    _stream.Write(header, 0, 4);
                    _stream.Write(payload, 0, payload.Length);
                    _stream.Flush();
                }
            }
            catch (Exception ex)
            {
                Print("_3_BayesianBridge_v71: Send failed: " + ex.Message);
            }
        }

        private void SendTradeClosed(string orderId, string side,
                                      double entryPrice, double exitPrice,
                                      double pnl, DateTime fillTime, int qty,
                                      bool isChain)
        {
            string json = "{"
                + Q("type") + ":" + Q("TRADE_CLOSED") + ","
                + Q("order_id") + ":" + Q(orderId) + ","
                + Q("side") + ":" + Q(side) + ","
                + Q("entry_price") + ":" + D2S(entryPrice) + ","
                + Q("exit_price") + ":" + D2S(exitPrice) + ","
                + Q("pnl") + ":" + D2S(pnl) + ","
                + Q("fill_time") + ":" + D2S(ToUnixSeconds(fillTime)) + ","
                + Q("qty") + ":" + qty + ","
                + Q("is_chain") + ":" + (isChain ? "true" : "false")
                + "}";
            SendRawJson(json);
        }

        private void SendOrderAck(string orderId)
        {
            string json = "{"
                + Q("type") + ":" + Q("ORDER_ACK") + ","
                + Q("order_id") + ":" + Q(orderId) + ","
                + Q("server_time") + ":" + D2S(ToUnixSeconds(DateTime.UtcNow))
                + "}";
            SendRawJson(json);
        }

        private void SendOrderRejected(string orderId, string reason)
        {
            string json = "{"
                + Q("type") + ":" + Q("ORDER_STATUS") + ","
                + Q("order_id") + ":" + Q(orderId) + ","
                + Q("status") + ":" + Q("Rejected") + ","
                + Q("reason") + ":" + Q(reason)
                + "}";
            SendRawJson(json);
        }

        private void SendPositionSnapshot()
        {
            if (_account == null) return;

            try
            {
                Position pos = FindPosition();
                int qty = 0;
                double avgPrice = 0;

                if (pos != null)
                {
                    if (pos.MarketPosition == MarketPosition.Long)
                        qty = pos.Quantity;
                    else if (pos.MarketPosition == MarketPosition.Short)
                        qty = -pos.Quantity;
                    avgPrice = pos.AveragePrice;
                }

                string json = "{"
                    + Q("type") + ":" + Q("POSITION") + ","
                    + Q("instrument") + ":" + Q(_targetInstr.FullName) + ","
                    + Q("qty") + ":" + qty + ","
                    + Q("avg_price") + ":" + D2S(avgPrice)
                    + "}";
                SendRawJson(json);
            }
            catch (Exception ex)
            {
                Print("_3_BayesianBridge_v71: Position snapshot failed: " + ex.Message);
            }
        }

        private void SendHistoryBuffer()
        {
            List<string> snapshot;
            lock (_barLock)
            {
                snapshot = new List<string>(_allBars);
            }

            foreach (string json in snapshot)
                SendRawJson(json);

            // Tell Python that the historical dump is complete
            string done = "{"
                + Q("type") + ":" + Q("HISTORY_DONE") + ","
                + Q("bar_count") + ":" + snapshot.Count
                + "}";
            SendRawJson(done);

            Print("_3_BayesianBridge_v71: Sent " + snapshot.Count + " historical bars to client");
        }

        private void SendHistoryBufferFrom(double afterTimestamp)
        {
            // Delta sync: only send bars with timestamp > afterTimestamp
            List<string> snapshot;
            lock (_barLock)
            {
                snapshot = new List<string>(_allBars);
            }

            int sent = 0;
            foreach (string json in snapshot)
            {
                // Fast timestamp extraction: find "timestamp": and parse the value
                int tsIdx = json.IndexOf("\"timestamp\":");
                if (tsIdx < 0) { SendRawJson(json); sent++; continue; }

                int valStart = tsIdx + 12; // length of "timestamp":
                int valEnd = json.IndexOf(',', valStart);
                if (valEnd < 0) valEnd = json.IndexOf('}', valStart);
                if (valEnd < 0) { SendRawJson(json); sent++; continue; }

                double barTs;
                if (double.TryParse(json.Substring(valStart, valEnd - valStart),
                    System.Globalization.NumberStyles.Any,
                    System.Globalization.CultureInfo.InvariantCulture, out barTs))
                {
                    if (barTs <= afterTimestamp)
                        continue; // skip — Python already has this bar
                }

                SendRawJson(json);
                sent++;
            }

            // Send HISTORY_DONE with the delta count
            string done = "{"
                + Q("type") + ":" + Q("HISTORY_DONE") + ","
                + Q("bar_count") + ":" + sent
                + "}";
            SendRawJson(done);

            Print("_3_BayesianBridge_v71: Delta sync — sent " + sent + "/" + snapshot.Count
                + " bars (after ts=" + afterTimestamp + ")");
        }

        private void SendAccountUpdate()
        {
            if (_account == null) return;

            try
            {
                double cashValue      = _account.Get(AccountItem.CashValue, Currency.UsDollar);
                double realizedPnl    = _account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                double unrealizedPnl  = _account.Get(AccountItem.UnrealizedProfitLoss, Currency.UsDollar);

                string json = "{"
                    + Q("type") + ":" + Q("ACCOUNT_UPDATE") + ","
                    + Q("cash_value") + ":" + D2S(cashValue) + ","
                    + Q("realized_pnl") + ":" + D2S(realizedPnl) + ","
                    + Q("unrealized_pnl") + ":" + D2S(unrealizedPnl) + ","
                    + Q("net_liquidation") + ":" + D2S(cashValue + unrealizedPnl) + ","
                    + Q("timestamp") + ":" + D2S(ToUnixSeconds(DateTime.UtcNow))
                    + "}";
                SendRawJson(json);
            }
            catch (Exception ex)
            {
                Print("_3_BayesianBridge_v71: Account update failed: " + ex.Message);
            }
        }

        // ── Position Lookup ───────────────────────────────────────────

        private Position FindPosition()
        {
            if (_account == null || _targetInstr == null) return null;
            foreach (Position p in _account.Positions)
            {
                if (p.Instrument.FullName == _targetInstr.FullName)
                    return p;
            }
            return null;
        }

        // ── JSON Helpers (no external dependencies) ──────────────────

        /// <summary>Quote a string for JSON: "value"</summary>
        private static string Q(string s)
        {
            if (s == null) return "null";
            return "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
        }

        /// <summary>Double to string with invariant culture (no locale comma issues)</summary>
        private static string D2S(double v)
        {
            return v.ToString(CultureInfo.InvariantCulture);
        }

        /// <summary>Parse flat JSON object into string dictionary. Handles quoted
        /// strings, numbers, and booleans. No nesting support needed.</summary>
        private static Dictionary<string, string> ParseSimpleJson(string json)
        {
            var dict = new Dictionary<string, string>();
            if (string.IsNullOrEmpty(json)) return dict;

            // Strip outer braces
            json = json.Trim();
            if (json.StartsWith("{")) json = json.Substring(1);
            if (json.EndsWith("}"))   json = json.Substring(0, json.Length - 1);

            int i = 0;
            while (i < json.Length)
            {
                // Skip whitespace/commas
                while (i < json.Length && (json[i] == ' ' || json[i] == ',' || json[i] == '\n'
                       || json[i] == '\r' || json[i] == '\t'))
                    i++;
                if (i >= json.Length) break;

                // Read key
                string key = ReadJsonValue(json, ref i);
                if (key == null) break;

                // Skip colon
                while (i < json.Length && (json[i] == ' ' || json[i] == ':'))
                    i++;

                // Read value
                string val = ReadJsonValue(json, ref i);
                if (val == null) break;

                dict[key] = val;
            }
            return dict;
        }

        private static string ReadJsonValue(string json, ref int i)
        {
            while (i < json.Length && json[i] == ' ') i++;
            if (i >= json.Length) return null;

            if (json[i] == '"')
            {
                // Quoted string
                i++; // skip opening quote
                var sb = new StringBuilder();
                while (i < json.Length && json[i] != '"')
                {
                    if (json[i] == '\\' && i + 1 < json.Length)
                    {
                        i++;
                        sb.Append(json[i]);
                    }
                    else
                    {
                        sb.Append(json[i]);
                    }
                    i++;
                }
                if (i < json.Length) i++; // skip closing quote
                return sb.ToString();
            }
            else
            {
                // Unquoted: number, bool, null
                int start = i;
                while (i < json.Length && json[i] != ',' && json[i] != '}'
                       && json[i] != ' ' && json[i] != '\n')
                    i++;
                return json.Substring(start, i - start);
            }
        }

        private static string GetVal(Dictionary<string, string> d, string key, string fallback)
        {
            string v;
            return d.TryGetValue(key, out v) ? v : fallback;
        }

        private static int GetIntVal(Dictionary<string, string> d, string key, int fallback)
        {
            string v;
            if (!d.TryGetValue(key, out v)) return fallback;
            int result;
            return int.TryParse(v, out result) ? result : fallback;
        }

        // ── Utilities ─────────────────────────────────────────────────

        private static int GetBarsPeriodSeconds(BarsPeriod bp)
        {
            switch (bp.BarsPeriodType)
            {
                case BarsPeriodType.Second: return bp.Value;
                case BarsPeriodType.Minute: return bp.Value * 60;
                case BarsPeriodType.Day:    return bp.Value * 86400;
                case BarsPeriodType.Week:   return bp.Value * 604800;
                default: return 1;  // Tick, Volume, Range — treat as 1s
            }
        }

        private static string GetTFLabel(int secs)
        {
            if (secs >= 86400) return (secs / 86400) + "D";
            if (secs >= 3600)  return (secs / 3600) + "h";
            if (secs >= 60)    return (secs / 60) + "m";
            return secs + "s";
        }

        private static double ToUnixSeconds(DateTime dt)
        {
            return (dt.ToUniversalTime() - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc))
                .TotalSeconds;
        }
    }
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private _3_BayesianBridge_v71[] cache_3_BayesianBridge_v71;
		public _3_BayesianBridge_v71 _3_BayesianBridge_v71(int port, string accountName, int domLevels, string targetInstrument)
		{
			return _3_BayesianBridge_v71(Input, port, accountName, domLevels, targetInstrument);
		}

		public _3_BayesianBridge_v71 _3_BayesianBridge_v71(ISeries<double> input, int port, string accountName, int domLevels, string targetInstrument)
		{
			if (cache_3_BayesianBridge_v71 != null)
				for (int idx = 0; idx < cache_3_BayesianBridge_v71.Length; idx++)
					if (cache_3_BayesianBridge_v71[idx] != null && cache_3_BayesianBridge_v71[idx].Port == port && cache_3_BayesianBridge_v71[idx].AccountName == accountName && cache_3_BayesianBridge_v71[idx].DomLevels == domLevels && cache_3_BayesianBridge_v71[idx].TargetInstrument == targetInstrument && cache_3_BayesianBridge_v71[idx].EqualsInput(input))
						return cache_3_BayesianBridge_v71[idx];
			return CacheIndicator<_3_BayesianBridge_v71>(new _3_BayesianBridge_v71(){ Port = port, AccountName = accountName, DomLevels = domLevels, TargetInstrument = targetInstrument }, input, ref cache_3_BayesianBridge_v71);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators._3_BayesianBridge_v71 _3_BayesianBridge_v71(int port, string accountName, int domLevels, string targetInstrument)
		{
			return indicator._3_BayesianBridge_v71(Input, port, accountName, domLevels, targetInstrument);
		}

		public Indicators._3_BayesianBridge_v71 _3_BayesianBridge_v71(ISeries<double> input , int port, string accountName, int domLevels, string targetInstrument)
		{
			return indicator._3_BayesianBridge_v71(input, port, accountName, domLevels, targetInstrument);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators._3_BayesianBridge_v71 _3_BayesianBridge_v71(int port, string accountName, int domLevels, string targetInstrument)
		{
			return indicator._3_BayesianBridge_v71(Input, port, accountName, domLevels, targetInstrument);
		}

		public Indicators._3_BayesianBridge_v71 _3_BayesianBridge_v71(ISeries<double> input , int port, string accountName, int domLevels, string targetInstrument)
		{
			return indicator._3_BayesianBridge_v71(input, port, accountName, domLevels, targetInstrument);
		}
	}
}

#endregion
