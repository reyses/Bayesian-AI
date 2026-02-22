// =============================================================================
// BayesianBridge — NinjaTrader 8 NinjaScript Indicator
//
// PURPOSE: TCP server inside NT8 that bridges to the Python live trading engine.
//   - Streams completed 15s OHLCV bars to Python
//   - Receives PLACE_ORDER / CLOSE_POSITION / CANCEL_ORDER from Python
//   - Sends FILL / ORDER_STATUS / POSITION messages back to Python
//
// INSTALLATION:
//   1. Copy this file to: Documents\NinjaTrader 8\bin\Custom\Indicators\
//   2. In NT8: Tools > NinjaScript Editor > right-click > Compile
//   3. Add indicator to a 15-second MNQ chart
//   4. Start the Python live engine: python -m live.launcher
//
// PROTOCOL: Length-prefixed JSON over TCP (port 5199)
//   Wire format: [4 bytes: uint32 big-endian payload length][N bytes: UTF-8 JSON]
//
// IMPORTANT: This is a REFERENCE implementation. Test thoroughly on Sim101
// before any live deployment.
// =============================================================================

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
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
        [Display(Name = "Account Name", Description = "Trading account (Sim101 for paper)", Order = 2, GroupName = "Bridge")]
        public string AccountName { get; set; }

        // ── Internal State ────────────────────────────────────────────
        private TcpListener  _listener;
        private TcpClient    _client;
        private NetworkStream _stream;
        private Thread        _serverThread;
        private Thread        _readThread;
        private volatile bool _running;
        private Account       _account;
        private readonly object _sendLock = new object();
        private readonly Queue<JObject> _inboundQueue = new Queue<JObject>();

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
            }
            else if (State == State.Configure)
            {
                // Nothing extra needed
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
                    Print($"BayesianBridge: Account '{AccountName}' not found!");
                    return;
                }

                // Subscribe to execution events
                _account.OrderUpdate    += OnOrderUpdate;
                _account.ExecutionUpdate += OnExecutionUpdate;
                _account.PositionUpdate  += OnPositionUpdate;

                // Start TCP server
                _running = true;
                _serverThread = new Thread(ServerLoop) { IsBackground = true };
                _serverThread.Start();

                Print($"BayesianBridge: Started on port {Port}, account={AccountName}");
            }
            else if (State == State.Terminated)
            {
                _running = false;

                if (_account != null)
                {
                    _account.OrderUpdate     -= OnOrderUpdate;
                    _account.ExecutionUpdate  -= OnExecutionUpdate;
                    _account.PositionUpdate   -= OnPositionUpdate;
                }

                try { _client?.Close(); }   catch { }
                try { _listener?.Stop(); }  catch { }

                Print("BayesianBridge: Stopped");
            }
        }

        // ── Bar Updates ───────────────────────────────────────────────

        protected override void OnBarUpdate()
        {
            // Only send on completed bars (not tick-by-tick)
            if (CurrentBars[0] < 1 || IsFirstTickOfBar == false)
                return;

            // Use previous bar (just completed)
            var msg = new JObject
            {
                ["type"]       = "BAR",
                ["instrument"] = Instrument.FullName,
                ["timestamp"]  = ToUnixSeconds(Time[1]),
                ["open"]       = Open[1],
                ["high"]       = High[1],
                ["low"]        = Low[1],
                ["close"]      = Close[1],
                ["volume"]     = Volume[1],
                ["bar_period_s"] = BarsPeriod.Value,
            };
            SendMessage(msg);
        }

        // ── Account Event Handlers ────────────────────────────────────

        private void OnOrderUpdate(object sender, OrderEventArgs e)
        {
            var msg = new JObject
            {
                ["type"]     = "ORDER_STATUS",
                ["order_id"] = e.Order.Name,
                ["status"]   = e.Order.OrderState.ToString(),
            };
            SendMessage(msg);
        }

        private void OnExecutionUpdate(object sender, ExecutionEventArgs e)
        {
            if (e.Execution == null) return;

            var msg = new JObject
            {
                ["type"]       = "FILL",
                ["order_id"]   = e.Order.Name,
                ["side"]       = e.Execution.MarketPosition == MarketPosition.Long ? "BUY" : "SELL",
                ["qty"]        = e.Execution.Quantity,
                ["fill_price"] = e.Execution.Price,
                ["fill_time"]  = ToUnixSeconds(e.Execution.Time),
                ["commission"] = e.Execution.Commission,
            };
            SendMessage(msg);
        }

        private void OnPositionUpdate(object sender, PositionEventArgs e)
        {
            if (e.Position == null) return;

            int qty = 0;
            if (e.Position.MarketPosition == MarketPosition.Long)
                qty = e.Position.Quantity;
            else if (e.Position.MarketPosition == MarketPosition.Short)
                qty = -e.Position.Quantity;

            var msg = new JObject
            {
                ["type"]           = "POSITION",
                ["instrument"]     = e.Position.Instrument.FullName,
                ["qty"]            = qty,
                ["avg_price"]      = e.Position.AveragePrice,
                ["unrealized_pnl"] = e.Position.GetUnrealizedProfitLoss(
                    PerformanceUnit.Currency, Close[0]),
            };
            SendMessage(msg);
        }

        // ── TCP Server ────────────────────────────────────────────────

        private void ServerLoop()
        {
            try
            {
                _listener = new TcpListener(IPAddress.Loopback, Port);
                _listener.Start();
                Print($"BayesianBridge: Listening on 127.0.0.1:{Port}");

                while (_running)
                {
                    // Accept one client at a time
                    if (!_listener.Pending())
                    {
                        Thread.Sleep(100);
                        continue;
                    }

                    _client = _listener.AcceptTcpClient();
                    _stream = _client.GetStream();
                    Print("BayesianBridge: Python client connected");

                    // Send CONNECTED message
                    var connMsg = new JObject
                    {
                        ["type"]    = "CONNECTED",
                        ["account"] = AccountName,
                    };
                    SendMessage(connMsg);

                    // Send current position snapshot
                    SendPositionSnapshot();

                    // Start read thread for inbound messages
                    _readThread = new Thread(ReadLoop) { IsBackground = true };
                    _readThread.Start();

                    // Wait until client disconnects
                    while (_running && _client != null && _client.Connected)
                    {
                        // Process any queued inbound commands
                        ProcessInboundQueue();
                        Thread.Sleep(50);
                    }

                    Print("BayesianBridge: Python client disconnected");
                }
            }
            catch (SocketException ex)
            {
                if (_running)
                    Print($"BayesianBridge: Server error: {ex.Message}");
            }
        }

        private void ReadLoop()
        {
            try
            {
                while (_running && _stream != null && _client.Connected)
                {
                    // Read 4-byte length header
                    byte[] header = ReadExactly(4);
                    if (header == null) break;

                    int length = (header[0] << 24) | (header[1] << 16)
                               | (header[2] << 8)  | header[3];

                    if (length <= 0 || length > 1048576) break;  // safety

                    // Read payload
                    byte[] payload = ReadExactly(length);
                    if (payload == null) break;

                    string json = Encoding.UTF8.GetString(payload);
                    JObject msg = JObject.Parse(json);

                    lock (_inboundQueue)
                    {
                        _inboundQueue.Enqueue(msg);
                    }
                }
            }
            catch (Exception ex)
            {
                if (_running)
                    Print($"BayesianBridge: Read error: {ex.Message}");
            }
        }

        private byte[] ReadExactly(int count)
        {
            byte[] buffer = new byte[count];
            int offset = 0;
            while (offset < count)
            {
                int read = _stream.Read(buffer, offset, count - offset);
                if (read == 0) return null;  // disconnected
                offset += read;
            }
            return buffer;
        }

        // ── Inbound Command Processing ────────────────────────────────

        private void ProcessInboundQueue()
        {
            List<JObject> commands;
            lock (_inboundQueue)
            {
                if (_inboundQueue.Count == 0) return;
                commands = new List<JObject>(_inboundQueue);
                _inboundQueue.Clear();
            }

            foreach (var cmd in commands)
            {
                string msgType = cmd["type"]?.ToString() ?? "";

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
                        Print($"BayesianBridge: SUBSCRIBE from Python — "
                            + $"instrument={cmd["instrument"]}, "
                            + $"bar_period_s={cmd["bar_period_s"]}");
                        break;
                    case "HEARTBEAT":
                        // Respond with server heartbeat
                        SendMessage(new JObject
                        {
                            ["type"]        = "HEARTBEAT",
                            ["server_time"] = ToUnixSeconds(DateTime.UtcNow),
                        });
                        break;
                    default:
                        Print($"BayesianBridge: Unknown command: {msgType}");
                        break;
                }
            }
        }

        private void HandlePlaceOrder(JObject cmd)
        {
            string orderId    = cmd["order_id"]?.ToString() ?? "BAY_UNK";
            string side       = cmd["side"]?.ToString() ?? "BUY";
            int    qty        = cmd["qty"]?.ToObject<int>() ?? 1;
            string instrument = cmd["instrument"]?.ToString() ?? "";

            Print($"BayesianBridge: PLACE_ORDER {side} {qty} {instrument} id={orderId}");

            // Submit market order via Account API
            try
            {
                Order order = _account.CreateOrder(
                    Instrument,
                    side == "BUY" ? OrderAction.Buy : OrderAction.SellShort,
                    OrderType.Market,
                    TimeInForce.Gtc,
                    qty,
                    0, 0,      // price, stopPrice
                    "", "",    // oco, signal
                    orderId,   // name
                    null       // custom
                );
                _account.Submit(new[] { order });
            }
            catch (Exception ex)
            {
                Print($"BayesianBridge: Order submit failed: {ex.Message}");
            }
        }

        private void HandleClosePosition(JObject cmd)
        {
            Print("BayesianBridge: CLOSE_POSITION");

            try
            {
                // Find current position and flatten it
                var pos = _account.Positions.FindByInstrument(Instrument);
                if (pos != null && pos.MarketPosition != MarketPosition.Flat)
                {
                    OrderAction action = pos.MarketPosition == MarketPosition.Long
                        ? OrderAction.Sell : OrderAction.BuyToCover;

                    Order closeOrder = _account.CreateOrder(
                        Instrument,
                        action,
                        OrderType.Market,
                        TimeInForce.Gtc,
                        pos.Quantity,
                        0, 0,
                        "", "",
                        "BAY_CLOSE",
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
                Print($"BayesianBridge: Close position failed: {ex.Message}");
            }
        }

        private void HandleCancelOrder(JObject cmd)
        {
            string orderId = cmd["order_id"]?.ToString() ?? "";
            Print($"BayesianBridge: CANCEL_ORDER {orderId}");

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
                Print($"BayesianBridge: Order {orderId} not found or not cancellable");
            }
            catch (Exception ex)
            {
                Print($"BayesianBridge: Cancel failed: {ex.Message}");
            }
        }

        // ── Outbound Messaging ────────────────────────────────────────

        private void SendMessage(JObject msg)
        {
            if (_stream == null || _client == null || !_client.Connected)
                return;

            try
            {
                string json    = msg.ToString(Formatting.None);
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
                Print($"BayesianBridge: Send failed: {ex.Message}");
            }
        }

        private void SendPositionSnapshot()
        {
            if (_account == null) return;

            try
            {
                var pos = _account.Positions.FindByInstrument(Instrument);
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

                SendMessage(new JObject
                {
                    ["type"]       = "POSITION",
                    ["instrument"] = Instrument.FullName,
                    ["qty"]        = qty,
                    ["avg_price"]  = avgPrice,
                });
            }
            catch (Exception ex)
            {
                Print($"BayesianBridge: Position snapshot failed: {ex.Message}");
            }
        }

        // ── Utilities ─────────────────────────────────────────────────

        private static double ToUnixSeconds(DateTime dt)
        {
            return (dt.ToUniversalTime() - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc))
                .TotalSeconds;
        }
    }
}
