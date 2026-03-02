// =============================================================================
// BayesianBridge — NinjaTrader 8 NinjaScript Indicator
//
// PURPOSE: TCP server inside NT8 that bridges to the Python live trading engine.
//   - Streams completed bars (any TF, e.g. 1s) to Python
//   - Receives PLACE_ORDER / CLOSE_POSITION / CANCEL_ORDER from Python
//   - Sends FILL / ORDER_STATUS / POSITION messages back to Python
//
// INSTALLATION:
//   1. Copy this file to: Documents\NinjaTrader 8\bin\Custom\Indicators\
//   2. In NT8: Tools > NinjaScript Editor > right-click > Compile
//   3. Add indicator to a 1-second MNQ chart on your sim account
//   4. Start the Python live engine: python -m live.launcher
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
    public class BayesianBridge : Indicator
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

        // ── Internal State ────────────────────────────────────────────
        private TcpListener  _listener;
        private TcpClient    _client;
        private NetworkStream _stream;
        private Thread        _serverThread;
        private Thread        _readThread;
        private volatile bool _running;
        private Account       _account;
        private readonly object _sendLock = new object();
        private readonly Queue<Dictionary<string, string>> _inboundQueue
            = new Queue<Dictionary<string, string>>();

        // Map BarsInProgress index -> period label for the BAR message
        // Index 0 = primary chart (1s), indices 1-11 = added data series
        private string[] _barLabels;
        private int[]    _barPeriodSecs;

        // History buffer — every completed bar is stored here; dumped to
        // client on connect so the Python engine can bypass warmup.
        private readonly List<string> _allBars = new List<string>();
        private readonly object _barLock = new object();

        // DOM throttle — send at most every N ms to avoid flooding
        private DateTime _lastDomSend = DateTime.MinValue;
        private const int DOM_THROTTLE_MS = 250;

        // Latest best bid/ask (updated on every depth event, sent throttled)
        private double _bestBid, _bestAsk;
        private long   _bestBidSize, _bestAskSize;

        // ── NinjaScript Lifecycle ─────────────────────────────────────

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "TCP bridge to Bayesian-AI Python live engine";
                Name        = "BayesianBridge";
                IsOverlay   = true;
                Port        = 5199;
                AccountName = "Sim101";
                DomLevels   = 5;
                MaximumBarsLookBack = MaximumBarsLookBack.Infinite;
                Calculate           = Calculate.OnBarClose;
            }
            else if (State == State.Configure)
            {
                // Add all timeframes the engine needs (12 TFs total).
                // Index 0 = primary chart (1s). Indices 1-11 below.
                // Order matches TF_HIERARCHY: 15s,30s,1m,2m,3m,5m,15m,30m,1h,4h,1D
                AddDataSeries(BarsPeriodType.Second, 15);   // idx 1: 15s
                AddDataSeries(BarsPeriodType.Second, 30);   // idx 2: 30s
                AddDataSeries(BarsPeriodType.Minute, 1);    // idx 3: 1m
                AddDataSeries(BarsPeriodType.Minute, 2);    // idx 4: 2m
                AddDataSeries(BarsPeriodType.Minute, 3);    // idx 5: 3m
                AddDataSeries(BarsPeriodType.Minute, 5);    // idx 6: 5m
                AddDataSeries(BarsPeriodType.Minute, 15);   // idx 7: 15m
                AddDataSeries(BarsPeriodType.Minute, 30);   // idx 8: 30m
                AddDataSeries(BarsPeriodType.Minute, 60);   // idx 9: 1h
                AddDataSeries(BarsPeriodType.Minute, 240);  // idx 10: 4h
                AddDataSeries(BarsPeriodType.Day, 1);       // idx 11: 1D

                _barLabels = new string[] {
                    "1s", "15s", "30s", "1m", "2m", "3m",
                    "5m", "15m", "30m", "1h", "4h", "1D"
                };
                _barPeriodSecs = new int[] {
                    1, 15, 30, 60, 120, 180,
                    300, 900, 1800, 3600, 14400, 86400
                };
            }
            else if (State == State.DataLoaded)
            {
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
                    Print("BayesianBridge: Account '" + AccountName + "' not found!");
                    return;
                }

                // Subscribe to execution events
                _account.OrderUpdate     += OnOrderUpdate;
                _account.ExecutionUpdate += OnExecutionUpdate;
                _account.PositionUpdate  += OnPositionUpdate;

                // Start TCP server
                _running = true;
                _serverThread = new Thread(ServerLoop) { IsBackground = true };
                _serverThread.Start();

                Print("BayesianBridge: Started on port " + Port + ", account=" + AccountName);
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

                Print("BayesianBridge: Stopped");
            }
        }

        // ── Bar Updates ───────────────────────────────────────────────

        protected override void OnBarUpdate()
        {
            int idx = BarsInProgress;
            if (_barLabels == null || idx >= _barLabels.Length)
                return;
            if (CurrentBars[idx] < 1 || IsFirstTickOfBar == false)
                return;

            // Build bar JSON for whichever TF just completed
            string json = "{"
                + Q("type") + ":" + Q("BAR") + ","
                + Q("instrument") + ":" + Q(Instrument.FullName) + ","
                + Q("tf") + ":" + Q(_barLabels[idx]) + ","
                + Q("bar_period_s") + ":" + _barPeriodSecs[idx] + ","
                + Q("timestamp") + ":" + D2S(ToUnixSeconds(Times[idx][1])) + ","
                + Q("open") + ":" + D2S(Opens[idx][1]) + ","
                + Q("high") + ":" + D2S(Highs[idx][1]) + ","
                + Q("low") + ":" + D2S(Lows[idx][1]) + ","
                + Q("close") + ":" + D2S(Closes[idx][1]) + ","
                + Q("volume") + ":" + D2S(Volumes[idx][1])
                + "}";

            // Always buffer (for history dump on reconnect); also send
            // live if a client is connected.
            lock (_barLock) { _allBars.Add(json); }
            SendRawJson(json);
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
                + Q("instrument") + ":" + Q(Instrument.FullName) + ","
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
            string json = "{"
                + Q("type") + ":" + Q("ORDER_STATUS") + ","
                + Q("order_id") + ":" + Q(e.Order.Name) + ","
                + Q("status") + ":" + Q(e.Order.OrderState.ToString())
                + "}";
            SendRawJson(json);
        }

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
        }

        private void OnPositionUpdate(object sender, PositionEventArgs e)
        {
            if (e.Position == null) return;

            int qty = 0;
            if (e.Position.MarketPosition == MarketPosition.Long)
                qty = e.Position.Quantity;
            else if (e.Position.MarketPosition == MarketPosition.Short)
                qty = -e.Position.Quantity;

            string json = "{"
                + Q("type") + ":" + Q("POSITION") + ","
                + Q("instrument") + ":" + Q(e.Position.Instrument.FullName) + ","
                + Q("qty") + ":" + qty + ","
                + Q("avg_price") + ":" + D2S(e.Position.AveragePrice) + ","
                + Q("unrealized_pnl") + ":" + D2S(e.Position.GetUnrealizedProfitLoss(
                    PerformanceUnit.Currency, Close[0]))
                + "}";
            SendRawJson(json);
        }

        // ── TCP Server ────────────────────────────────────────────────

        private void ServerLoop()
        {
            try
            {
                _listener = new TcpListener(IPAddress.Loopback, Port);
                _listener.Start();
                Print("BayesianBridge: Listening on 127.0.0.1:" + Port);

                while (_running)
                {
                    if (!_listener.Pending())
                    {
                        Thread.Sleep(100);
                        continue;
                    }

                    _client = _listener.AcceptTcpClient();
                    _stream = _client.GetStream();
                    Print("BayesianBridge: Python client connected");

                    // Send CONNECTED message
                    string connJson = "{"
                        + Q("type") + ":" + Q("CONNECTED") + ","
                        + Q("account") + ":" + Q(AccountName)
                        + "}";
                    SendRawJson(connJson);

                    SendPositionSnapshot();
                    SendHistoryBuffer();

                    _readThread = new Thread(ReadLoop) { IsBackground = true };
                    _readThread.Start();

                    while (_running && _client != null && _client.Connected)
                    {
                        ProcessInboundQueue();
                        Thread.Sleep(50);
                    }

                    Print("BayesianBridge: Python client disconnected");
                }
            }
            catch (SocketException ex)
            {
                if (_running)
                    Print("BayesianBridge: Server error: " + ex.Message);
            }
        }

        private void ReadLoop()
        {
            try
            {
                while (_running && _stream != null && _client.Connected)
                {
                    byte[] header = ReadExactly(4);
                    if (header == null) break;

                    int length = (header[0] << 24) | (header[1] << 16)
                               | (header[2] << 8)  | header[3];

                    if (length <= 0 || length > 1048576) break;

                    byte[] payload = ReadExactly(length);
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
                    Print("BayesianBridge: Read error: " + ex.Message);
            }
        }

        private byte[] ReadExactly(int count)
        {
            byte[] buffer = new byte[count];
            int offset = 0;
            while (offset < count)
            {
                int read = _stream.Read(buffer, offset, count - offset);
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
                        Print("BayesianBridge: SUBSCRIBE — instrument="
                            + inst + ", bar_period_s=" + bps);
                        break;
                    case "HEARTBEAT":
                        string hb = "{"
                            + Q("type") + ":" + Q("HEARTBEAT") + ","
                            + Q("server_time") + ":" + D2S(ToUnixSeconds(DateTime.UtcNow))
                            + "}";
                        SendRawJson(hb);
                        break;
                    default:
                        Print("BayesianBridge: Unknown command: " + msgType);
                        break;
                }
            }
        }

        private void HandlePlaceOrder(Dictionary<string, string> cmd)
        {
            string orderId = GetVal(cmd, "order_id", "BAY_UNK");
            string side    = GetVal(cmd, "side", "BUY");
            int    qty     = GetIntVal(cmd, "qty", 1);

            Print("BayesianBridge: PLACE_ORDER " + side + " " + qty + " id=" + orderId);

            try
            {
                Order order = _account.CreateOrder(
                    Instrument,
                    side == "BUY" ? OrderAction.Buy : OrderAction.SellShort,
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
                Print("BayesianBridge: Order submit failed: " + ex.Message);
            }
        }

        private void HandleClosePosition(Dictionary<string, string> cmd)
        {
            Print("BayesianBridge: CLOSE_POSITION");

            try
            {
                Position pos = FindPosition();
                if (pos != null && pos.MarketPosition != MarketPosition.Flat)
                {
                    OrderAction action = pos.MarketPosition == MarketPosition.Long
                        ? OrderAction.Sell : OrderAction.BuyToCover;

                    Order closeOrder = _account.CreateOrder(
                        Instrument,
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
                    Print("BayesianBridge: No position to close");
                }
            }
            catch (Exception ex)
            {
                Print("BayesianBridge: Close position failed: " + ex.Message);
            }
        }

        private void HandleCancelOrder(Dictionary<string, string> cmd)
        {
            string orderId = GetVal(cmd, "order_id", "");
            Print("BayesianBridge: CANCEL_ORDER " + orderId);

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
                Print("BayesianBridge: Order " + orderId + " not found or not cancellable");
            }
            catch (Exception ex)
            {
                Print("BayesianBridge: Cancel failed: " + ex.Message);
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
                Print("BayesianBridge: Send failed: " + ex.Message);
            }
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
                    + Q("instrument") + ":" + Q(Instrument.FullName) + ","
                    + Q("qty") + ":" + qty + ","
                    + Q("avg_price") + ":" + D2S(avgPrice)
                    + "}";
                SendRawJson(json);
            }
            catch (Exception ex)
            {
                Print("BayesianBridge: Position snapshot failed: " + ex.Message);
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

            Print("BayesianBridge: Sent " + snapshot.Count + " historical bars to client");
        }

        // ── Position Lookup ───────────────────────────────────────────

        private Position FindPosition()
        {
            if (_account == null) return null;
            foreach (Position p in _account.Positions)
            {
                if (p.Instrument.FullName == Instrument.FullName)
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

        private static double ToUnixSeconds(DateTime dt)
        {
            return (dt.ToUniversalTime() - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc))
                .TotalSeconds;
        }
    }
}
