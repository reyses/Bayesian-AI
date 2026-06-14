// ============================================================================
// BayesianHistoryDumper — Dump NT8 chart bars to CSV for ATLAS pipeline
// ============================================================================
// Version: 2.4.2 (2026-06-13)
//
// CHANGELOG 2.4.2 (2026-06-13):
//   - FIX: Enabled AutoFlush on StreamWriters to prevent data loss and 64KB
//     truncation if NinjaTrader crashes or abruptly stops the script before
//     the Terminated state can gracefully flush the final buffer.
//
// CHANGELOG 2.4.0 (2026-06-13):
//   - DIAGNOSTICS for the NinjaScript Output window so a non-dumping run is
//     debuggable: a version banner on load; a print for every lifecycle state
//     (Configure / DataLoaded / Historical / Transition / Realtime / Terminated);
//     a per-series FIRST-BAR line (State + CurrentBar + bars-available + time —
//     reveals if a timeframe starts in REALTIME with no history to dump); file
//     writes wrapped in try/catch (surfaces path / OneDrive-lock errors instead
//     of failing silently); and a 0-BARS warning at the end for any timeframe
//     that never wrote. No change to output format or partitioning.
//
// CHANGELOG 2.3.0 (2026-06-13):
//   - SESSION-DAY partitioning: daily files now split on the CME session boundary
//     (17:00 America/Chicago reopen, DST-aware via TimeZoneInfo) instead of UTC
//     midnight — mirrors core_v2/sessions.py on the Databento/IS side. A "day"
//     file now holds ONE continuous session (Sun-eve reopen -> next 16:00 CT
//     close), so the maintenance halt sits at the file EDGE, never mid-file.
//     Monday's session = Sun 17:00 CT -> Mon 16:00 CT, labelled = Monday's date.
//     NOTE: not required before a capture — the Python consumer can re-partition
//     by session-day from the UTC ts — but keeps the raw CSV folders consistent.
//
// CHANGELOG 2.2.0 (2026-06-13):
//   - SAFETY: logs the chart's actual instrument (Instrument.FullName) at load
//     and WARNS if it doesn't match ContractLabel. Prevents dumping the WRONG
//     contract around a roll (the one real "wrong data" failure mode). Pure
//     logging — does NOT change what gets dumped, so it is safe to run as-is.
//
// CHANGELOG 2.1.0 (BREAKING -- path reorganization):
//   - Output structure now CONTRACT-FIRST instead of TF-first. New layout:
//       {OutputDirectory}\{ContractLabel}\1s\YYYY_MM_DD.csv
//       {OutputDirectory}\{ContractLabel}\1m\YYYY_MM_DD.csv
//       {OutputDirectory}\{ContractLabel}\1h\YYYY_MM_DD.csv
//       {OutputDirectory}\{ContractLabel}\1D\YYYY_MM_DD.csv
//   - Default OutputDirectory changed from DATA\ATLAS_NT8 to DATA\RAW_NT8 to
//     cleanly separate raw CSV from the parquet-pipeline ATLAS_NT8 dir.
//   - Multi-contract friendly: each contract gets its own subfolder, easier
//     to extend to rollover contracts (MNQ_03-26, MNQ_06-26, MNQ_09-26, ...).
//   - For UPGRADING existing CSV layout (DATA\ATLAS_NT8\{tf}\MNQ_06-26\), see
//     the one-shot mv done 2026-05-18 (commit log).
//
// CHANGELOG 2.0.0 (2026-04-27):
//   - Now dumps 1s, 1m, 1h, 1D simultaneously from ONE chart via AddDataSeries.
//     Previous v1.0.0 required opening 4 separate charts and running once per
//     timeframe. New version: open any chart, add indicator once, done.
//   - Each TF maintains independent writer + day-rotation + bar counter state.
//   - Per-TF "COMPLETE" progress logging.
//   - Primary chart timeframe is IGNORED — pick whatever's convenient (1s gives
//     fastest data load; daily chart loads slowest but is fine).
//
// CHANGELOG 1.0.0 (2026-04-10):
//   - Initial single-TF dumper. Required user to open separate charts per TF.
//
// PURPOSE:
//   Exports NT8 chart bars to CSV files (one CSV per day per timeframe).
//   The Python pipeline reads these CSVs, converts to parquet, and uses
//   them as the ATLAS_NT8 data source for feature building + analysis.
//   Ensures Python-side analysis matches NT8-feed bars exactly.
//
// SETUP:
//   1. Copy to: Documents\NinjaTrader 8\bin\Custom\Indicators\
//   2. Tools > NinjaScript Editor > Compile (F5)
//   3. Open ANY chart on MNQ 06-26 (1s recommended for fastest data load).
//      Set "load N days" high enough to cover desired range.
//   4. Add this indicator to the chart.
//   5. Wait for "COMPLETE — all timeframes" in Output window.
//   6. For rollover: switch chart to MNQ 09-26, set ContractLabel accordingly,
//      remove + re-add indicator. CSVs go to a new contract subfolder.
//
// OUTPUT:
//   {OutputDirectory}\{ContractLabel}\{tf}\YYYY_MM_DD.csv
//   Format: timestamp,open,high,low,close,volume (header + data rows, UTF-8)
//   Timestamps: UTC unix epoch seconds (bar OPEN time, Databento convention)
//
// PYTHON CONSUMER:
//   python tools/sourcing/convert_nt8_csv_to_parquet.py
//   - Reads DATA/RAW_NT8/{contract}/{tf}/*.csv
//   - Converts to DATA/ATLAS_NT8/{tf}/*.parquet (parquet stays flat, no contract subdir)
//   - 1m: shift ts +59 (bar-close convention)
//   - 1s: no shift, then rebin to 5s/15s/30s/5m/15m/30m/4h with NT8 alignment
//   - Validates rebin against existing parquet (byte-identical for boring TFs)
// ============================================================================

#region Using declarations
using System;
using System.IO;
using System.Globalization;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class BayesianHistoryDumper : Indicator
    {
        // ── TF definitions ──
        // BarsInProgress index -> (TF label, period seconds)
        // 0 = primary chart (ignored — user can pick any chart timeframe)
        // 1..4 = our explicit AddDataSeries calls
        private static readonly string[] TF_LABELS  = { null, "1s", "1m", "1h", "1D" };
        private static readonly int[]    TF_SECONDS = { 0,    1,    60,   3600, 86400 };
        private const int N_TFS = 4;  // = TF_LABELS.Length - 1
        private const string VERSION = "2.4.2";

        // Session-day boundary = CME equity-index reopen 17:00 America/Chicago
        // (DST-aware). Mirrors core_v2/sessions.py so NT8 CSVs partition exactly
        // like the Databento/IS side.
        private static readonly TimeZoneInfo CmeTz =
            TimeZoneInfo.FindSystemTimeZoneById("Central Standard Time");
        private const int SESSION_REOPEN_HOUR_CT = 17;

        // ── Per-TF state (indexed 1..N_TFS, slot 0 unused) ──
        private StreamWriter[] _writer        = new StreamWriter[N_TFS + 1];
        private string[]       _currentDay    = new string[N_TFS + 1];
        private int[]          _barsWritten   = new int[N_TFS + 1];
        private int[]          _daysWritten   = new int[N_TFS + 1];
        private bool[]         _tfComplete    = new bool[N_TFS + 1];
        private bool[]         _firstBarSeen  = new bool[N_TFS + 1];  // for the first-bar diagnostic
        private DateTime[]     _firstUtc      = new DateTime[N_TFS + 1];  // first dumped bar (UTC), for range log
        private DateTime[]     _lastUtc       = new DateTime[N_TFS + 1];  // last  dumped bar (UTC), for range log

        // ── Properties (visible in NT8 UI) ──
        [NinjaScriptProperty]
        [Display(Name = "Output Directory", GroupName = "Settings", Order = 0,
                 Description = "BASE directory for CSV output. v2.1.0 layout: {OutputDirectory}/{ContractLabel}/{TF}/. " +
                               "Default: DATA/RAW_NT8")]
        public string OutputDirectory { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Contract Label", GroupName = "Settings", Order = 1,
                 Description = "Contract subdir name (e.g. MNQ_06-26, MNQ_09-26). " +
                               "Each contract gets its own subfolder; TFs nested inside.")]
        public string ContractLabel { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Start Date", GroupName = "Settings", Order = 2,
                 Description = "Only dump bars from this date onwards (YYYY_MM_DD). Empty = all available history.")]
        public string StartDate { get; set; }

        // ── Lifecycle ──

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description              = "Dump NT8 chart bars (1s/1m/1h/1D) to CSV for Bayesian-AI ATLAS_NT8 pipeline";
                Name                     = "BayesianHistoryDumper";
                IsOverlay                = true;
                IsSuspendedWhileInactive = false;
                MaximumBarsLookBack      = MaximumBarsLookBack.Infinite;
                Calculate                = Calculate.OnBarClose;

                // Defaults — user can override in UI
                OutputDirectory = @"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\DATA\RAW_NT8";
                ContractLabel   = "MNQ_06-26";
                StartDate       = "";   // empty = dump all available history
            }
            else if (State == State.Configure)
            {
                Print("BayesianHistoryDumper v" + VERSION + ": Configure — adding 1s/1m/1h/1D data series.");
                // Add the four explicit data series we need to dump.
                // Order MUST match TF_LABELS / TF_SECONDS indices 1..4.
                AddDataSeries(BarsPeriodType.Second, 1);   // BarsInProgress = 1 -> 1s
                AddDataSeries(BarsPeriodType.Minute, 1);   // BarsInProgress = 2 -> 1m
                AddDataSeries(BarsPeriodType.Minute, 60);  // BarsInProgress = 3 -> 1h
                AddDataSeries(BarsPeriodType.Day,    1);   // BarsInProgress = 4 -> 1D
            }
            else if (State == State.DataLoaded)
            {
                Print("==================================================");
                Print("BayesianHistoryDumper v" + VERSION + ": DataLoaded — creating folders + starting dump.");
                // Pre-create TF subfolders so folder structure exists even if a TF has zero bars
                for (int i = 1; i <= N_TFS; i++)
                {
                    string folder = TfFolder(i);
                    Directory.CreateDirectory(folder);
                    Print("BayesianHistoryDumper: " + TF_LABELS[i] + " -> " + folder);
                }
                Print("BayesianHistoryDumper: ContractLabel=" + ContractLabel
                      + ", StartDate=" + (StartDate.Length > 0 ? StartDate : "(none)"));

                // SAFETY: confirm the chart instrument matches ContractLabel so we
                // never silently dump the WRONG contract (easy mistake around a roll).
                string chartInstr = Instrument.FullName;              // e.g. "MNQ 06-26"
                string expected   = ContractLabel.Replace('_', ' ');  // "MNQ_06-26" -> "MNQ 06-26"
                Print("BayesianHistoryDumper: chart instrument = " + chartInstr);
                if (!string.Equals(chartInstr, expected, StringComparison.OrdinalIgnoreCase))
                    Print("BayesianHistoryDumper: *** WARNING *** chart instrument '" + chartInstr
                          + "' != ContractLabel '" + expected + "'. You may be dumping the WRONG contract — "
                          + "set the chart to " + expected + " (or fix ContractLabel) before trusting this dump.");

                Print("BayesianHistoryDumper: Processing chart bars across 4 timeframes...");
            }
            else if (State == State.Historical)
            {
                Print("BayesianHistoryDumper: entering HISTORICAL — dumping past bars now.");
            }
            else if (State == State.Transition)
            {
                Print("BayesianHistoryDumper: entering TRANSITION (historical -> realtime).");
            }
            else if (State == State.Realtime)
            {
                Print("BayesianHistoryDumper: entering REALTIME — each timeframe finishes as its series hits realtime.");
            }
            else if (State == State.Terminated)
            {
                for (int i = 1; i <= N_TFS; i++)
                    CloseWriter(i);
                Print("BayesianHistoryDumper: Final summary:");
                for (int i = 1; i <= N_TFS; i++)
                    Print("  " + TF_LABELS[i] + ": " + _barsWritten[i] + " bars, "
                          + FmtUtc(_firstUtc[i]) + " -> " + FmtUtc(_lastUtc[i]) + " (" + _daysWritten[i] + " days)"
                          + (_barsWritten[i] == 0
                             ? "   *** 0 BARS — never dumped (series started in realtime, no data, wrong chart, or Days-to-load too low?) ***"
                             : ""));
            }
        }

        // ── Bar Processing ──

        protected override void OnBarUpdate()
        {
            int bip = BarsInProgress;

            // Ignore primary chart series — we only consume our explicit AddDataSeries.
            if (bip < 1 || bip > N_TFS) return;

            // FIRST-BAR diagnostic per series — shows whether this TF starts in HISTORICAL
            // (will dump) or REALTIME (nothing to dump), and how many bars are loaded.
            if (!_firstBarSeen[bip])
            {
                _firstBarSeen[bip] = true;
                Print("BayesianHistoryDumper: " + TF_LABELS[bip] + " first bar — State=" + State
                      + ", CurrentBar=" + CurrentBars[bip]
                      + ", bars available=" + BarsArray[bip].Count
                      + ", time=" + Times[bip][0].ToString("yyyy-MM-dd HH:mm"));
            }

            if (_tfComplete[bip]) return;

            // Detect realtime transition for THIS series — historical dump for this TF is done.
            if (State == State.Realtime)
            {
                CloseWriter(bip);
                _tfComplete[bip] = true;
                Print("BayesianHistoryDumper: " + TF_LABELS[bip] + " COMPLETE — "
                      + _barsWritten[bip] + " bars, " + FmtUtc(_firstUtc[bip]) + " -> " + FmtUtc(_lastUtc[bip])
                      + " (" + _daysWritten[bip] + " days)");
                AnnounceAllCompleteIfDone();
                return;
            }

            // Get bar OPEN time (Databento convention).
            // NT8 Times[bip][0] is bar CLOSE time on OnBarClose; subtract bar period.
            DateTime barCloseTime = Times[bip][0];
            int barPeriodS = TF_SECONDS[bip];
            DateTime openTime = barCloseTime.AddSeconds(-barPeriodS);
            double ts = ToUnixSeconds(openTime);
            DateTime utc = openTime.ToUniversalTime();
            // Session-day label (CME 17:00 CT boundary, DST-aware) — NOT UTC/local midnight.
            DateTime ct = TimeZoneInfo.ConvertTimeFromUtc(utc, CmeTz);
            string day = (ct.Hour >= SESSION_REOPEN_HOUR_CT ? ct.Date.AddDays(1) : ct.Date)
                         .ToString("yyyy_MM_dd");

            // Skip bars before StartDate
            if (StartDate.Length > 0 && day.CompareTo(StartDate) < 0)
                return;

            try
            {
                // Day boundary — rotate file for THIS timeframe
                if (day != _currentDay[bip])
                {
                    CloseWriter(bip);
                    string filePath = Path.Combine(TfFolder(bip), day + ".csv");
                    _writer[bip] = new StreamWriter(filePath, false, System.Text.Encoding.UTF8, 65536);
                    _writer[bip].AutoFlush = true; // FORCE flush to disk immediately
                    _writer[bip].WriteLine("timestamp,open,high,low,close,volume");
                    _currentDay[bip] = day;
                    _daysWritten[bip]++;
                }

                _writer[bip].WriteLine(
                    D2S(ts) + ","
                    + D2S(Opens[bip][0])  + "," + D2S(Highs[bip][0]) + ","
                    + D2S(Lows[bip][0])   + "," + D2S(Closes[bip][0]) + ","
                    + D2S(Volumes[bip][0]));
                _barsWritten[bip]++;
                if (_barsWritten[bip] == 1) _firstUtc[bip] = utc;
                _lastUtc[bip] = utc;
            }
            catch (Exception ex)
            {
                Print("BayesianHistoryDumper: *** WRITE ERROR *** " + TF_LABELS[bip] + " day=" + day
                      + " : " + ex.Message + "  (path / permission / OneDrive lock? check OutputDirectory)");
                _tfComplete[bip] = true;   // stop repeating the same error on every bar
            }

            // Progress logging — every 50K bars on 1s, every 5K bars on 1m, every 500 on 1h, every 50 on 1D
            int progressEvery = (bip == 1) ? 50000 : (bip == 2) ? 5000 : (bip == 3) ? 500 : 50;
            if (_barsWritten[bip] % progressEvery == 0)
                Print("BayesianHistoryDumper: " + TF_LABELS[bip] + " " + _barsWritten[bip]
                      + " bars, " + _daysWritten[bip] + " days...");
        }

        // ── Helpers ──

        private string TfFolder(int bip)
        {
            // v2.1.0: contract-first hierarchy
            // {OutputDirectory}/{ContractLabel}/{TF}/
            return Path.Combine(Path.Combine(OutputDirectory, ContractLabel), TF_LABELS[bip]);
        }

        private void CloseWriter(int bip)
        {
            if (_writer[bip] != null)
            {
                _writer[bip].Flush();
                _writer[bip].Close();
                _writer[bip].Dispose();
                _writer[bip] = null;
            }
        }

        private void AnnounceAllCompleteIfDone()
        {
            for (int i = 1; i <= N_TFS; i++)
                if (!_tfComplete[i]) return;
            Print("BayesianHistoryDumper: COMPLETE — all timeframes done");
            Print("BayesianHistoryDumper: Run `python tools/atlas_nt8_rebuild.py` to convert CSVs to parquet");
        }

        /// <summary>Format a UTC DateTime for the log, or "(none)" if unset.</summary>
        private static string FmtUtc(DateTime dt)
        {
            return dt == DateTime.MinValue ? "(none)" : dt.ToString("yyyy-MM-dd HH:mm") + " UTC";
        }

        /// <summary>Double to string with invariant culture (no locale comma issues).</summary>
        private static string D2S(double v)
        {
            return v.ToString(CultureInfo.InvariantCulture);
        }

        /// <summary>Convert DateTime to UTC unix epoch seconds.</summary>
        private static double ToUnixSeconds(DateTime dt)
        {
            return (dt.ToUniversalTime() - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc))
                .TotalSeconds;
        }
    }
}
