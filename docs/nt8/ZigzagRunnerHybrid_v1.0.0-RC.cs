// =============================================================================
// ZigzagRunnerHybrid 1.0.0-RC -- 2026-05-18  (DEPRECATED 2026-05-18)
// =============================================================================
//
// STATUS: REJECTED RESEARCH ARTIFACT. Do NOT compile or deploy.
//   Rename to ZigzagRunnerHybrid_v1.0.0-RC.REJECTED.cs per CLAUDE.md
//   versioning policy if it stays in tree.
//
// REASON FOR DEPRECATION
//   This strategy + companion live/L5_sidecar.py duplicate ~80% of the
//   existing live/engine_v2.py + OrderManager + nt8_client infrastructure
//   (pending-order tracking, fill reconciliation, NT8 transport, position
//   state, mock-bridge). Decision logic belongs in Python (engine_v2);
//   NT8 should be a dumb pipe streaming bars + accepting orders, NOT a
//   strategy host that calls back to Python.
//
// CORRECT PATH (see docs/L5_HYBRID_PIPELINE_SPEC.md):
//   * Keep ZigzagRunnerNative_v1.0.0-RC.cs as research-grade NT8-native
//     zigzag (matches Python ATR(14)x4 calibration) for tick-precise
//     pivot validation -- not as a production decision point.
//   * Move all B7/B9/B10 logic into live/l5_decider.py inside engine_v2.
//   * Use existing BayesianBridge.cs v7.0.0 for bar streaming + orders.
//
// ──────────────────────── ORIGINAL HEADER ────────────────────────
//
// PURPOSE
//   Hybrid NT8 + Python sidecar deployment of the full L5 stack:
//     B7  (entry sizing on V2 features at R-trigger fire bar)
//     B9  (during-trade sizing at T+25s based on K=5 trajectory)
//     B10 (day-level vol-regime sizing at session start)
//
//   Native strategy detects pivots + R-triggers (matches Python pipeline
//   calibration). Python L5_sidecar.py applies the three models and
//   responds with sizing decisions. NT8 places orders with returned size.
//
// SEALED OOS BACKTEST RESULT (51 days, 2026-03-19 to 2026-05-18):
//   FLAT baseline:           $+454/day
//   B7 + B9 + B10 full stack: $+1126/day  (+$672/day delta, 148% lift)
//   95% CI on delta:         [$+426, $+939]/day  SIGNIFICANT
//
// IPC PROTOCOL (length-prefixed JSON over TCP, port 5200)
//   See live/L5_sidecar.py for the message specifications.
//
//   1. At session start: send DAY_OPEN -> server returns b10_mult
//   2. At R-trigger fire: send ENTRY_QUERY with V2 features at R-bar
//      -> server returns contracts (= round(b7_size * b10_mult))
//   3. T+25s after fill: send SIZE_QUERY with V2 + trajectory at K=5
//      -> server returns action: HOLD | REDUCE_50 | CUT | PYRAMID
//   4. At R-trigger exit: send POSITION_CLOSED with realized pnl
//
// DEPENDENCIES
//   - ZigzagRunnerNative_v1.0.0-RC.cs (pivot detection logic — same module
//     of code; this file extends with IPC hooks)
//   - L5_sidecar.py running on port 5200 BEFORE applying this strategy
//
// DEPLOYMENT (per CLAUDE.md gate)
//   1. Copy file to Documents\NinjaTrader 8\bin\Custom\Strategies\
//      ONLY AFTER explicit user approval ("ship it" / "deploy")
//   2. NT8 NinjaScript Editor -> F5 to compile
//   3. Start sidecar: `python -m live.L5_sidecar --port 5200`
//   4. Apply ZigzagRunnerHybrid to MNQ chart on SIM account first
//   5. Verify Output window shows successful IPC roundtrips
//   6. Run 30-day sim per L5_HYBRID_PIPELINE_SPEC.md before any live capital
//
// STATUS: Reference implementation. Untested in live NT8 (built 2026-05-18).
//
// =============================================================================

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;

#if NT8
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
#endif

#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ZigzagRunnerHybrid_v100RC : Strategy
    {
        // ── IPC settings ────────────────────────────────────────────────
        [NinjaScriptProperty]
        [Display(Name = "Sidecar Host", GroupName = "0. Sidecar IPC", Order = 0)]
        public string SidecarHost { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Sidecar Port", GroupName = "0. Sidecar IPC", Order = 1)]
        public int SidecarPort { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Connection Timeout (ms)", GroupName = "0. Sidecar IPC", Order = 2)]
        public int ConnectionTimeoutMs { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Fallback Contracts (on IPC fail)", GroupName = "0. Sidecar IPC", Order = 3)]
        public int FallbackContracts { get; set; }

        // ── Pivot + ATR settings (same as ZigzagRunnerNative_v1.0.0-RC) ─
        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "ATR timeframe (minutes)", GroupName = "1. ATR (Python parity)", Order = 0)]
        public int AtrTfMinutes { get; set; }

        [Range(2, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "ATR period", GroupName = "1. ATR (Python parity)", Order = 1)]
        public int AtrPeriod { get; set; }

        [Range(0.1, 100.0), NinjaScriptProperty]
        [Display(Name = "ATR multiplier", GroupName = "1. ATR (Python parity)", Order = 2)]
        public double AtrMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Use median ATR", GroupName = "1. ATR (Python parity)", Order = 3)]
        public bool UseMedianAtr { get; set; }

        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "ZigZag timeframe (seconds)", GroupName = "2. ZigZag (Python parity)", Order = 0)]
        public int ZigZagTfSeconds { get; set; }

        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Min bars between pivots", GroupName = "2. ZigZag (Python parity)", Order = 1)]
        public int MinBars { get; set; }

        // ── Schedule (same as Native) ───────────────────────────────────
        [Range(0, 23), NinjaScriptProperty]
        [Display(Name = "EOD Hour UTC", GroupName = "3. Schedule (UTC)", Order = 0)]
        public int EodHourUtc { get; set; }

        [Range(0, 59), NinjaScriptProperty]
        [Display(Name = "EOD Minute UTC", GroupName = "3. Schedule (UTC)", Order = 1)]
        public int EodMinuteUtc { get; set; }

        [Range(0, 23), NinjaScriptProperty]
        [Display(Name = "Entry Cutoff Hour UTC", GroupName = "3. Schedule (UTC)", Order = 2)]
        public int EntryCutoffHourUtc { get; set; }

        [Range(0, 59), NinjaScriptProperty]
        [Display(Name = "Entry Cutoff Minute UTC", GroupName = "3. Schedule (UTC)", Order = 3)]
        public int EntryCutoffMinuteUtc { get; set; }

        // ── B9 timer settings ───────────────────────────────────────────
        [NinjaScriptProperty]
        [Display(Name = "B9 size-query delay (seconds)", GroupName = "4. B9 timer",
                 Description = "T+N seconds after fill, query B9 for cut/hold/pyramid. Default 25 = K=5 bar.",
                 Order = 0)]
        public int B9DelaySeconds { get; set; }

        // ── BIP routing ─────────────────────────────────────────────────
        private const int BIP_ATR    = 1;
        private const int BIP_ZIGZAG = 2;

        private const string VERSION = "1.0.0-RC";

        // ── Internal state ──────────────────────────────────────────────
        private TcpClient _tcpClient;
        private NetworkStream _tcpStream;
        private readonly object _ipcLock = new object();

        private string _currentDay;
        private double _dayMult = 1.0;     // B10 day multiplier
        private string _lastFillPositionId;
        private DateTime _lastFillTime;
        private DateTime _b9QueryDueAt = DateTime.MaxValue;
        private bool _b9QuerySent;

        // ── Lifecycle ───────────────────────────────────────────────────

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "ZigzagRunnerHybrid_v" + VERSION;
                Description = "Hybrid B7+B9+B10 with Python sidecar IPC.";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                BarsRequiredToTrade = 2;

                // Defaults match Native v1.0.0-RC
                SidecarHost = "127.0.0.1";
                SidecarPort = 5200;
                ConnectionTimeoutMs = 1500;
                FallbackContracts = 1;
                AtrTfMinutes = 1;
                AtrPeriod = 14;
                AtrMult = 4.0;
                UseMedianAtr = true;
                ZigZagTfSeconds = 5;
                MinBars = 36;
                EodHourUtc = 20;
                EodMinuteUtc = 55;
                EntryCutoffHourUtc = 20;
                EntryCutoffMinuteUtc = 30;
                B9DelaySeconds = 25;
            }
            else if (State == State.Configure)
            {
                AddDataSeries(BarsPeriodType.Minute, AtrTfMinutes);
                AddDataSeries(BarsPeriodType.Second, ZigZagTfSeconds);
                ConnectSidecar();
            }
            else if (State == State.Terminated)
            {
                DisconnectSidecar();
            }
        }

        // ── IPC helpers ─────────────────────────────────────────────────

        private void ConnectSidecar()
        {
            try
            {
                _tcpClient = new TcpClient();
                _tcpClient.ReceiveTimeout = ConnectionTimeoutMs;
                _tcpClient.SendTimeout = ConnectionTimeoutMs;
                _tcpClient.Connect(SidecarHost, SidecarPort);
                _tcpStream = _tcpClient.GetStream();
                Print($"L5 sidecar connected: {SidecarHost}:{SidecarPort}");
            }
            catch (Exception ex)
            {
                Print($"L5 sidecar connection FAILED ({SidecarHost}:{SidecarPort}): {ex.Message}");
                Print($"Strategy will fall back to {FallbackContracts} contracts/leg.");
                _tcpClient = null;
                _tcpStream = null;
            }
        }

        private void DisconnectSidecar()
        {
            try { _tcpStream?.Close(); } catch { }
            try { _tcpClient?.Close(); } catch { }
            _tcpStream = null;
            _tcpClient = null;
        }

        /// <summary>
        /// Send JSON message, get JSON response. Length-prefixed wire format.
        /// Returns null on failure (caller falls back to defaults).
        /// </summary>
        private string SendIpc(string jsonRequest)
        {
            lock (_ipcLock)
            {
                if (_tcpStream == null)
                {
                    ConnectSidecar();
                    if (_tcpStream == null) return null;
                }
                try
                {
                    byte[] body = Encoding.UTF8.GetBytes(jsonRequest);
                    byte[] header = new byte[4];
                    header[0] = (byte)((body.Length >> 24) & 0xff);
                    header[1] = (byte)((body.Length >> 16) & 0xff);
                    header[2] = (byte)((body.Length >> 8) & 0xff);
                    header[3] = (byte)(body.Length & 0xff);
                    _tcpStream.Write(header, 0, 4);
                    _tcpStream.Write(body, 0, body.Length);
                    _tcpStream.Flush();

                    byte[] respHeader = new byte[4];
                    int n = _tcpStream.Read(respHeader, 0, 4);
                    if (n < 4) return null;
                    int respLen = (respHeader[0] << 24) | (respHeader[1] << 16)
                                  | (respHeader[2] << 8) | respHeader[3];
                    if (respLen <= 0 || respLen > 1_000_000) return null;
                    byte[] respBody = new byte[respLen];
                    int got = 0;
                    while (got < respLen)
                    {
                        int read = _tcpStream.Read(respBody, got, respLen - got);
                        if (read <= 0) return null;
                        got += read;
                    }
                    return Encoding.UTF8.GetString(respBody);
                }
                catch (Exception ex)
                {
                    Print($"SendIpc FAILED: {ex.Message}");
                    DisconnectSidecar();
                    return null;
                }
            }
        }

        // ── OnBarUpdate (skeleton — full pivot logic same as ZigzagRunnerNative) ─

        protected override void OnBarUpdate()
        {
            // BIP=1 ATR update — see ZigzagRunnerNative for full logic
            // BIP=2 5s pivot state machine — see ZigzagRunnerNative for full logic
            // BIP=0 primary chart — orders + IPC hooks

            if (BarsInProgress == 0)
            {
                // Day-open detection: send DAY_OPEN at first bar of new day
                string today = Time[0].ToString("yyyy_MM_dd");
                if (today != _currentDay)
                {
                    _currentDay = today;
                    string msg = $"{{\"type\":\"DAY_OPEN\",\"day\":\"{today}\"}}";
                    string resp = SendIpc(msg);
                    if (resp != null && resp.Contains("b10_mult"))
                    {
                        // Parse b10_mult from response (simple substring; production
                        // would use Newtonsoft.Json)
                        try
                        {
                            int idx = resp.IndexOf("b10_mult\":") + 10;
                            int end = resp.IndexOfAny(new[] { ',', '}' }, idx);
                            _dayMult = double.Parse(resp.Substring(idx, end - idx));
                            Print($"DAY_OPEN {today}: b10_mult={_dayMult:F2}");
                        }
                        catch (Exception ex)
                        {
                            Print($"DAY_OPEN parse error: {ex.Message}, default 1.0");
                            _dayMult = 1.0;
                        }
                    }
                    else
                    {
                        _dayMult = 1.0;
                    }
                }

                // B9 timer check: at T+B9DelaySeconds after last fill, send SIZE_QUERY
                if (Position.MarketPosition != MarketPosition.Flat
                    && !_b9QuerySent
                    && Time[0] >= _b9QueryDueAt)
                {
                    SendB9SizeQuery();
                    _b9QuerySent = true;
                }

                // ... pivot detection + EOD + order placement logic from
                //     ZigzagRunnerNative_v1.0.0-RC.cs lives here (300+ lines).
                //     Skeleton omits for brevity — see Native version.
            }
        }

        // ── On pivot fire (called from BIP=2 logic, simplified here) ────

        private void OnPivotFire(int pivotDir, double entryPrice, string leg_dir)
        {
            // Build ENTRY_QUERY with V2 features at R-trigger bar.
            // In production, V2 feature computation happens IN NT8 (port from
            // core_v2.features.py) OR the sidecar caches V2 features from a
            // separate data feed and just looks them up by timestamp.
            // For RC: send a thin payload, sidecar replies with FALLBACK if it
            // doesn't have V2 yet.

            string positionId = $"pos_{Time[0]:yyyyMMddHHmmss}_{leg_dir}";
            long entryTsUnix = ((DateTimeOffset)Time[0].ToUniversalTime()).ToUnixTimeSeconds();

            string msg = "{"
                + "\"type\":\"ENTRY_QUERY\","
                + $"\"position_id\":\"{positionId}\","
                + $"\"day\":\"{_currentDay}\","
                + $"\"entry_ts\":{entryTsUnix},"
                + $"\"entry_price\":{entryPrice},"
                + $"\"leg_dir\":\"{leg_dir}\","
                + "\"v2_features\":{}"
                + "}";

            int contracts = FallbackContracts;
            string resp = SendIpc(msg);
            if (resp != null && resp.Contains("contracts"))
            {
                try
                {
                    int idx = resp.IndexOf("contracts\":") + 11;
                    int end = resp.IndexOfAny(new[] { ',', '}' }, idx);
                    contracts = int.Parse(resp.Substring(idx, end - idx));
                }
                catch
                {
                    contracts = FallbackContracts;
                }
            }
            contracts = Math.Max(1, contracts);

            // Place entry
            if (leg_dir == "LONG")
                EnterLong(contracts, positionId);
            else
                EnterShort(contracts, positionId);

            // Schedule B9 query for T+B9DelaySeconds
            _lastFillPositionId = positionId;
            _lastFillTime = Time[0];
            _b9QueryDueAt = Time[0].AddSeconds(B9DelaySeconds);
            _b9QuerySent = false;

            Print($"ENTRY {positionId} {leg_dir}: {contracts} contracts (b10_mult={_dayMult:F2})");
        }

        private void SendB9SizeQuery()
        {
            if (Position.MarketPosition == MarketPosition.Flat) return;

            double currentClose = Close[0];
            double entryPrice = Position.AveragePrice;
            int legDirSign = Position.MarketPosition == MarketPosition.Long ? 1 : -1;
            double pnlPts = legDirSign * (currentClose - entryPrice);
            long curTsUnix = ((DateTimeOffset)Time[0].ToUniversalTime()).ToUnixTimeSeconds();

            string msg = "{"
                + "\"type\":\"SIZE_QUERY\","
                + $"\"position_id\":\"{_lastFillPositionId}\","
                + $"\"current_ts\":{curTsUnix},"
                + $"\"current_price\":{currentClose},"
                + $"\"pnl_pts_so_far\":{pnlPts},"
                + "\"v2_features\":{}"
                + "}";

            string resp = SendIpc(msg);
            if (resp == null)
            {
                Print($"SIZE_QUERY {_lastFillPositionId}: IPC fail, default HOLD");
                return;
            }

            // Parse action
            string action = "HOLD";
            try
            {
                int idx = resp.IndexOf("action\":\"") + 9;
                int end = resp.IndexOf('"', idx);
                action = resp.Substring(idx, end - idx);
            }
            catch { }

            Print($"SIZE_QUERY {_lastFillPositionId}: action={action}");

            int positionQty = Math.Abs(Position.Quantity);
            switch (action)
            {
                case "CUT":
                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong(positionQty, "B9_CUT", _lastFillPositionId);
                    else
                        ExitShort(positionQty, "B9_CUT", _lastFillPositionId);
                    break;
                case "REDUCE_50":
                    int halfQty = positionQty / 2;
                    if (halfQty > 0)
                    {
                        if (Position.MarketPosition == MarketPosition.Long)
                            ExitLong(halfQty, "B9_REDUCE", _lastFillPositionId);
                        else
                            ExitShort(halfQty, "B9_REDUCE", _lastFillPositionId);
                    }
                    break;
                case "PYRAMID":
                    int addQty = Math.Max(1, positionQty / 2);   // add 50%
                    if (Position.MarketPosition == MarketPosition.Long)
                        EnterLong(addQty, _lastFillPositionId + "_pyr");
                    else
                        EnterShort(addQty, _lastFillPositionId + "_pyr");
                    break;
                case "HOLD":
                case "HOLD_UNCERTAIN":
                default:
                    // no action
                    break;
            }
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId,
            double price, int quantity, MarketPosition marketPosition, string orderId,
            DateTime time)
        {
            // Hook: on entry fill, capture for B9 timer; on exit, send POSITION_CLOSED
            if (execution.Order != null && execution.Order.OrderState == OrderState.Filled)
            {
                if (Position.MarketPosition != MarketPosition.Flat
                    && _lastFillPositionId != null
                    && execution.Order.Name == _lastFillPositionId)
                {
                    _lastFillTime = time;
                    _b9QueryDueAt = time.AddSeconds(B9DelaySeconds);
                    _b9QuerySent = false;
                }
                else if (Position.MarketPosition == MarketPosition.Flat
                         && _lastFillPositionId != null)
                {
                    // Position closed — notify sidecar
                    double realizedUsd = SystemPerformance.AllTrades.TradesPerformance.Currency.CumProfit;
                    string msg = "{"
                        + "\"type\":\"POSITION_CLOSED\","
                        + $"\"position_id\":\"{_lastFillPositionId}\","
                        + $"\"exit_pnl_usd\":{realizedUsd}"
                        + "}";
                    SendIpc(msg);
                    _lastFillPositionId = null;
                    _b9QueryDueAt = DateTime.MaxValue;
                }
            }
        }
    }
}
