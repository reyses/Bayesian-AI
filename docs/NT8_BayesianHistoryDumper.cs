// ============================================================================
// BayesianHistoryDumper — Dump NT8 chart bars to CSV for ATLAS pipeline
// ============================================================================
// Version: 1.0.0 (2026-04-10)
//
// PURPOSE:
//   Exports ALL bars from an NT8 chart to CSV files (one per day).
//   The Python pipeline reads these CSVs, converts to parquet, and uses
//   them as the ATLAS data source for feature building + training.
//   This ensures training data matches live data exactly (same source).
//
// SETUP:
//   1. Copy to: Documents\NinjaTrader 8\bin\Custom\Indicators\
//   2. Tools > NinjaScript Editor > Compile
//   3. Open a NEW chart: MNQ 06-26, 5 Second bars, load 180 days
//   4. Add this indicator to the chart
//   5. Wait for "COMPLETE" in Output window
//   6. For rollover: repeat with MNQ 03-25 chart, set ContractLabel accordingly
//
// OUTPUT:
//   DATA/ATLAS_NT8/5s/{ContractLabel}/YYYY_MM_DD.csv
//   Format: timestamp,open,high,low,close,volume (header + data rows)
//   Timestamps: UTC unix epoch seconds (float64)
//
// PYTHON CONVERTER:
//   python tools/convert_nt8_atlas.py --backup
// ============================================================================

#region Using declarations
using System;
using System.IO;
using System.Globalization;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class BayesianHistoryDumper : Indicator
    {
        // ── State ──
        private StreamWriter _writer;
        private string       _currentDay   = "";
        private int          _barsWritten  = 0;
        private int          _daysWritten  = 0;
        private bool         _dumpComplete = false;
        private string       _outputPath   = "";

        // ── Properties (visible in NT8 UI) ──
        [NinjaScriptProperty]
        [Display(Name = "Output Directory", GroupName = "Settings", Order = 0,
                 Description = "Root directory for CSV output")]
        public string OutputDirectory { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Contract Label", GroupName = "Settings", Order = 1,
                 Description = "Subdirectory name (e.g. MNQ_06-26)")]
        public string ContractLabel { get; set; }

        // ── Lifecycle ──

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description         = "Dump chart bars to CSV for Bayesian-AI ATLAS pipeline";
                Name                = "BayesianHistoryDumper";
                IsOverlay           = true;
                IsSuspendedWhileInactive = false;
                MaximumBarsLookBack = MaximumBarsLookBack.Infinite;
                Calculate           = Calculate.OnBarClose;

                // Defaults — user can override in UI
                OutputDirectory = @"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\DATA\ATLAS_NT8\5s";
                ContractLabel   = "MNQ_06-26";
            }
            else if (State == State.DataLoaded)
            {
                // Create output directory
                _outputPath = Path.Combine(OutputDirectory, ContractLabel);
                Directory.CreateDirectory(_outputPath);
                Print("BayesianHistoryDumper: Output -> " + _outputPath);
                Print("BayesianHistoryDumper: Processing chart bars...");
            }
            else if (State == State.Terminated)
            {
                CloseWriter();
                if (_barsWritten > 0)
                    Print("BayesianHistoryDumper: Final — "
                          + _barsWritten + " bars across " + _daysWritten + " days");
            }
        }

        // ── Bar Processing ──

        protected override void OnBarUpdate()
        {
            if (_dumpComplete) return;

            // Only process primary series
            if (BarsInProgress != 0) return;

            // Detect realtime transition — historical dump is done
            if (State == State.Realtime)
            {
                CloseWriter();
                _dumpComplete = true;
                Print("BayesianHistoryDumper: COMPLETE — "
                      + _barsWritten + " bars across " + _daysWritten + " days");
                Print("BayesianHistoryDumper: Output at " + _outputPath);
                return;
            }

            // Get bar data — Time[0] is bar OPEN time in NT8 (OnBarClose context)
            // Databento convention: timestamp = bar CLOSE time
            // Fix: add bar period to align with Databento
            DateTime barTime = Time[0];
            int barPeriodS = (int)BarsPeriod.Value; // e.g. 5 for 5-second bars
            double ts = ToUnixSeconds(barTime) + barPeriodS;

            // Use close-time for day grouping (matches Databento)
            DateTime closeTime = barTime.AddSeconds(barPeriodS);
            string day = closeTime.ToString("yyyy_MM_dd");

            // Day boundary — rotate file
            if (day != _currentDay)
            {
                CloseWriter();
                string filePath = Path.Combine(_outputPath, day + ".csv");
                _writer = new StreamWriter(filePath, false, System.Text.Encoding.UTF8, 65536);
                _writer.WriteLine("timestamp,open,high,low,close,volume");
                _currentDay = day;
                _daysWritten++;
            }
            _writer.WriteLine(
                D2S(ts) + ","
                + D2S(Open[0]) + "," + D2S(High[0]) + ","
                + D2S(Low[0]) + "," + D2S(Close[0]) + ","
                + D2S(Volume[0]));
            _barsWritten++;

            // Progress every 50K bars
            if (_barsWritten % 50000 == 0)
                Print("BayesianHistoryDumper: " + _barsWritten + " bars, " + _daysWritten + " days...");
        }

        // ── Helpers ──

        private void CloseWriter()
        {
            if (_writer != null)
            {
                _writer.Flush();
                _writer.Close();
                _writer.Dispose();
                _writer = null;
            }
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
