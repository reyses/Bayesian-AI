// =============================================================================
// BayesianCompanion 1.0.0-RC -- 2026-05-20
// =============================================================================
//
// NT8 display-only companion for the Bayesian-AI L5 live engine.
//
// PURPOSE
//   Mirrors the Python live dashboard ONTO the NT8 chart. Whatever the
//   Python engine is "thinking" -- zigzag extreme, R-trigger threshold,
//   B7/B9/B10 state, skipped signals, trades -- is drawn on your MNQ
//   chart so you don't have to watch two screens.
//
// ARCHITECTURE -- display only, ZERO logic.
//   The Python engine writes its L5 state to a flat file every 5s bar:
//       live/state/l5_overlay.txt   (atomic tmp+replace write)
//   This indicator POLLS that file and draws it. It does NOT compute
//   zigzag / B7 / B9 / anything -- that would duplicate Python logic and
//   drift. Python is the brain; this is a windshield.
//
// INSTALLATION (RC -- deploy on explicit approval)
//   1. Copy to: Documents\NinjaTrader 8\bin\Custom\Indicators\
//   2. NT8: NinjaScript Editor > Compile
//   3. Add BayesianCompanion to your MNQ chart (any TF).
//   4. Set "Overlay File" to the absolute path of live/state/l5_overlay.txt.
//
// WHAT IT DRAWS
//   - amber horizontal line  : zigzag running extreme
//   - cyan horizontal line   : R-trigger threshold (price must reach here
//                              for a pivot / entry to fire)
//   - bottom-left text block : r_price / dir / B10 mode / B7 / B9 / skips
//   - hollow amber triangle  : a skipped R-trigger signal (down=short,
//                              up=long would-be trade)
//   - green/red arrow        : trade entry / exit
//
// =============================================================================

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.IO;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class BayesianCompanion : Indicator
    {
        // ── Settings ──────────────────────────────────────────────────
        [NinjaScriptProperty]
        [Display(Name = "Overlay File", Description = "Absolute path to live/state/l5_overlay.txt",
                 Order = 1, GroupName = "Companion")]
        public string OverlayFile { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Poll Seconds", Description = "How often to re-read the overlay file",
                 Order = 2, GroupName = "Companion")]
        public double PollSeconds { get; set; }

        private const string COMPANION_VERSION = "1.0.0-RC";

        // ── Internal state ────────────────────────────────────────────
        private DateTime _lastRead = DateTime.MinValue;
        // Event timestamps already drawn (so each skip/trade is drawn once)
        private readonly HashSet<long> _drawnEvents = new HashSet<long>();

        // Unix-ts -> NT8 bar-time map, built from the bars this indicator
        // processes. Lets the zigzag line anchor pivots to the right bars
        // without any timezone conversion (the Python pivot ts and this
        // map are both derived from the same bridge ToUnixSeconds math).
        private readonly List<long>     _barUnix = new List<long>();
        private readonly List<DateTime> _barTime = new List<DateTime>();
        private int _lastMappedBar = -1;
        private int _lastZzSegCount = 0;

        private static readonly DateTime _epoch =
            new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Display-only mirror of the Bayesian-AI L5 Python engine";
                Name        = "BayesianCompanion";
                IsOverlay   = true;
                Calculate   = Calculate.OnEachTick;
                OverlayFile = @"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\live\state\l5_overlay.txt";
                PollSeconds = 1.0;
            }
        }

        // ── Refresh hooks: bar close + each tick (throttled) ─────────
        protected override void OnBarUpdate()
        {
            // Record this bar's (unix, time) once per bar so the zigzag
            // line can anchor Python pivot timestamps to chart bars.
            if (CurrentBar != _lastMappedBar && CurrentBar >= 0)
            {
                _lastMappedBar = CurrentBar;
                _barUnix.Add(ToUnix(Time[0]));
                _barTime.Add(Time[0]);
                // Cap memory -- a session is well under this.
                if (_barUnix.Count > 20000)
                {
                    _barUnix.RemoveRange(0, 5000);
                    _barTime.RemoveRange(0, 5000);
                }
            }
            RefreshOverlay();
        }

        protected override void OnMarketData(MarketDataEventArgs e)
        {
            if (e.MarketDataType == MarketDataType.Last)
                RefreshOverlay();
        }

        // ── Core: read the overlay file + draw ───────────────────────
        private void RefreshOverlay()
        {
            // Only draw in realtime -- the overlay reflects the live engine.
            if (State != State.Realtime)
                return;

            // Throttle file reads.
            if ((DateTime.Now - _lastRead).TotalSeconds < PollSeconds)
                return;
            _lastRead = DateTime.Now;

            string[] lines;
            try
            {
                if (!File.Exists(OverlayFile))
                    return;
                lines = File.ReadAllLines(OverlayFile);
            }
            catch
            {
                return;   // file mid-write or locked -- try again next poll
            }

            // ── Parse: key=value fields + SKIP/TRADE event lines ──────
            var kv = new Dictionary<string, string>();
            var events = new List<string[]>();
            foreach (string raw in lines)
            {
                string line = raw.Trim();
                if (line.Length == 0)
                    continue;
                if (line.StartsWith("SKIP|") || line.StartsWith("TRADE|"))
                {
                    events.Add(line.Split('|'));
                    continue;
                }
                int eq = line.IndexOf('=');
                if (eq > 0)
                    kv[line.Substring(0, eq)] = line.Substring(eq + 1);
            }

            DrawZigzag(kv);
            DrawLevels(kv);
            DrawStatusText(kv);
            DrawEvents(events);
        }

        // ── Zigzag line: connect confirmed pivots + live segment ─────
        // Mirrors what the engine sees -- straight segments pivot-to-pivot
        // (like NT8's built-in ZigZag), so you see the engine's swing
        // interpretation, not just the close.
        private void DrawZigzag(Dictionary<string, string> kv)
        {
            string raw = Get(kv, "zz_pivots", "");
            // Build the point list: confirmed pivots + the live extreme.
            var times = new List<DateTime>();
            var prices = new List<double>();
            if (raw.Length > 0)
            {
                foreach (string p in raw.Split('|'))
                {
                    int c = p.IndexOf(':');
                    if (c <= 0)
                        continue;
                    long pts  = ParseLongRaw(p.Substring(0, c), -1);
                    double px = ParseDoubleRaw(p.Substring(c + 1), 0);
                    if (pts <= 0 || px <= 0)
                        continue;
                    DateTime bt;
                    if (FindBarTime(pts, out bt))
                    {
                        times.Add(bt);
                        prices.Add(px);
                    }
                }
            }
            // Live segment endpoint: the running extreme (not yet a pivot).
            double extPx = ParseDouble(kv, "zz_extreme", 0);
            long   extTs = ParseLongRaw(Get(kv, "zz_extreme_ts", "0"), 0);
            if (extPx > 0 && extTs > 0)
            {
                DateTime bt;
                if (FindBarTime(extTs, out bt))
                {
                    times.Add(bt);
                    prices.Add(extPx);
                }
            }

            // Draw connected segments. Tag per segment so they update in
            // place each refresh.
            int seg = 0;
            for (int i = 0; i + 1 < times.Count; i++)
            {
                Draw.Line(this, "bc_zz_" + i, false,
                          times[i], prices[i], times[i + 1], prices[i + 1],
                          Brushes.DodgerBlue, DashStyleHelper.Solid, 2);
                seg++;
            }
            // Remove stale segment tags if the line got shorter.
            for (int i = seg; i < _lastZzSegCount; i++)
                RemoveDrawObject("bc_zz_" + i);
            _lastZzSegCount = seg;
        }

        // Nearest-bar lookup: Python pivot ts -> this chart's bar time.
        private bool FindBarTime(long ts, out DateTime result)
        {
            result = DateTime.MinValue;
            if (_barUnix.Count == 0)
                return false;
            // Linear scan for the closest unix value (bar lists are small
            // within a session; pivots are recent).
            int best = -1;
            long bestDiff = long.MaxValue;
            for (int i = _barUnix.Count - 1; i >= 0; i--)
            {
                long d = Math.Abs(_barUnix[i] - ts);
                if (d < bestDiff)
                {
                    bestDiff = d;
                    best = i;
                }
                // bars are time-ordered descending in this scan; once the
                // diff starts growing we've passed the closest.
                else if (d > bestDiff)
                {
                    break;
                }
            }
            if (best < 0)
                return false;
            result = _barTime[best];
            return true;
        }

        private static long ToUnix(DateTime dt)
        {
            return (long)(dt.ToUniversalTime() - _epoch).TotalSeconds;
        }

        // ── Zigzag extreme + R-trigger threshold lines ───────────────
        private void DrawLevels(Dictionary<string, string> kv)
        {
            double ext = ParseDouble(kv, "zz_extreme", 0);
            double thr = ParseDouble(kv, "zz_threshold", 0);

            if (ext > 0)
                Draw.HorizontalLine(this, "bc_extreme", ext, Brushes.Orange,
                                    DashStyleHelper.Dash, 1);
            if (thr > 0)
                Draw.HorizontalLine(this, "bc_rtrigger", thr, Brushes.Cyan,
                                    DashStyleHelper.Dash, 1);
        }

        // ── Bottom-left status text block ────────────────────────────
        private void DrawStatusText(Dictionary<string, string> kv)
        {
            string dir   = Get(kv, "zz_dir", "?").ToUpper();
            string mode  = Get(kv, "b10_mode", "normal").ToUpper();
            string rp    = Get(kv, "r_price", "0");
            string b7    = Get(kv, "b7", "--");
            string b9    = Get(kv, "b9", "--");
            string skips = Get(kv, "skip_count", "0");
            string pos   = Get(kv, "position", "flat").ToUpper();
            string pnl   = Get(kv, "day_pnl", "0");
            if (b7.Length == 0) b7 = "--";
            if (b9.Length == 0) b9 = "--";

            string text =
                "L5 ENGINE  (Python mirror)\n"
                + "r_price " + rp + "   dir " + dir + "   " + mode + "\n"
                + "B7 " + b7 + "   B9 " + b9 + "   skips " + skips + "\n"
                + "position " + pos + "   day $" + pnl;

            Draw.TextFixed(this, "bc_status", text, TextPosition.BottomLeft,
                           Brushes.White, new SimpleFont("Consolas", 11),
                           Brushes.Transparent, Brushes.Transparent, 0);
        }

        // ── Skip + trade markers (drawn once per event timestamp) ────
        private void DrawEvents(List<string[]> events)
        {
            foreach (string[] parts in events)
            {
                if (parts.Length < 4)
                    continue;
                string kind = parts[0];               // SKIP | TRADE
                long evTs   = ParseLongRaw(parts[1], -1);
                double px   = ParseDoubleRaw(parts[2], 0);
                string dir  = parts[3];

                if (evTs <= 0 || px <= 0)
                    continue;
                if (_drawnEvents.Contains(evTs))
                    continue;            // already drawn
                _drawnEvents.Add(evTs);

                // Anchor at the current bar (barsAgo=0). The companion polls
                // ~1s, Python writes per 5s bar, so a fresh event lands on
                // the current/last bar -- no timezone conversion needed.
                string tag = "bc_" + kind.ToLower() + "_" + evTs;
                if (kind == "SKIP")
                {
                    // hollow amber triangle, pointed the would-be trade way
                    if (dir == "short")
                        Draw.TriangleDown(this, tag, false, 0, px + 2 * TickSize,
                                          Brushes.Orange);
                    else
                        Draw.TriangleUp(this, tag, false, 0, px - 2 * TickSize,
                                        Brushes.Orange);
                }
                else if (kind == "TRADE")
                {
                    string mode = parts.Length > 4 ? parts[4] : "";
                    if (mode == "ENTRY")
                    {
                        if (dir == "long")
                            Draw.ArrowUp(this, tag, false, 0, px - 3 * TickSize,
                                         Brushes.LimeGreen);
                        else
                            Draw.ArrowDown(this, tag, false, 0, px + 3 * TickSize,
                                           Brushes.Red);
                    }
                    else  // EXIT
                    {
                        Draw.Diamond(this, tag, false, 0, px, Brushes.Gold);
                    }
                }
            }
        }

        // ── Parse helpers ────────────────────────────────────────────
        private static string Get(Dictionary<string, string> kv, string k, string dflt)
        {
            string v;
            return kv.TryGetValue(k, out v) ? v : dflt;
        }

        private static double ParseDouble(Dictionary<string, string> kv, string k, double dflt)
        {
            string v;
            if (kv.TryGetValue(k, out v))
                return ParseDoubleRaw(v, dflt);
            return dflt;
        }

        private static double ParseDoubleRaw(string v, double dflt)
        {
            double d;
            return double.TryParse(v, NumberStyles.Any,
                                   CultureInfo.InvariantCulture, out d) ? d : dflt;
        }

        private static long ParseLongRaw(string v, long dflt)
        {
            long l;
            return long.TryParse(v, NumberStyles.Any,
                                 CultureInfo.InvariantCulture, out l) ? l : dflt;
        }
    }
}
