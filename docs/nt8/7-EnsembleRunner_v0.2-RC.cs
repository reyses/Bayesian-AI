//
// 7-EnsembleRunner_v0.2-RC.cs
// -----------------------------------------------------------------------------
// NinjaTrader 8 STRATEGY: Ensemble entry combiner + R-trigger ride-only exit.
// v0.2-RC -- the DECISION CORE is now ported IN (v0.1 was a stub with 14 TODOs).
//
// ORIGIN (2026-07-18, research/nt8_port P2b): the 22 top-K generators, the frozen
// 27-col logistic combiner, the TMPL0 frozen codebook + P2 same-bar tie rule, the
// ±180s consensus, and the native R-trigger zigzag are the SAME code proven at
// BIT-EXACT parity in the C# harness (research/nt8_port/csharp). That core is
// carried here VERBATIM inside `namespace EnsembleV02Core` (the SHARED-CORE-V02
// region), down-levelled to C# 7.3 / .NET 4.8. The identical region is compiled by
// the V02ParityShim console driver, which re-runs it over the golden 20 days:
//   * 22 top-K stream fire-states : 100.000% (178,640 / 178,640 cells)
//   * governing entry decision    : 100.000% (8,120 / 8,120 bars; 913 entries)
//   * gov direction on entries    : 100.000% (913 / 913)
//   * compact combiner P          : max |dP| = 2.22e-16
//   * R-trigger zigzag leg+pivot  : 100.000%; pivot age/price bit-exact (0.0e0)
//   * shim out == harness out     : byte-identical, all 20 days
// (see research/nt8_port/reports/p2b_v02_parity.md)
//
// THE SHARED-CORE-V02 region is machine-injected from the single source of truth
//   research/nt8_port/csharp/v02/EnsembleCoreV02.region.cs
// via research/nt8_port/csharp/v02/assemble.py, and re-checked byte-for-byte by
// v02/verify_region.py. Do NOT hand-edit the region here -- edit the source + rerun.
//
// WHAT THIS IS: a mechanical manager (Architecture B, doc 129) -- NO cut logic.
//   Entry  : pooled 22-stream logistic combiner P >= frozen top-decile threshold
//            (0.7139834155227371). Side = the governing (max-P) stream's direction.
//   Exit   : R-trigger REVERSAL ONLY (ride-only, doc 107). A confirmed zigzag pivot
//            AGAINST the open position closes it. No fixed TP, no MFE cut, no trail.
//   Sizing : fixed 1 contract.
//   Guards : optional catastrophic stop (default OFF in SIM); flatten at 15:55 CT.
//
// STREAMING MODEL (see CHANGELOG + p2b report §Deviations): the proven core is a
// per-DAY batch. This strategy buffers the session's 5s bars and re-runs the core
// once per completed 1-minute bar over the day-so-far. Every generator is CAUSAL,
// so a minute's decision is FINAL once the ±180s consensus window has elapsed; the
// strategy therefore acts on a minute's entry only after that 180s settle. That
// makes the DECISION bit-identical to the golden; the only cost is a 3-bar (1m)
// action latency vs the golden timestamp -- flagged P2-8 (fill semantics).
//
// DEPLOY GATE: this is an -RC. It has NOT been NT8-compiled and NOTHING has been
// copied to Documents/NinjaTrader 8/bin/Custom/Strategies/. Promotion requires
// explicit per-revision user approval.
//
// VERSION: 0.2.0-RC
// -----------------------------------------------------------------------------
#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies.EnsembleV02Core;   // <- shared decision core
// NT8 compile fix: bare `Core` resolves to the enclosing NinjaTrader.Core assembly
// namespace before file-level usings are consulted (CS0234) -> alias the class.
using V02Core = NinjaTrader.NinjaScript.Strategies.EnsembleV02Core.Core;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class EnsembleRunner_v02 : Strategy
    {
        private const string VERSION = "0.2.0-RC";

        // ---- frozen decision constants (embedded in EnsembleV02Core.ModelData) ----
        private const double TICK = 0.25;                       // MNQ tick size (points)
        // Entry threshold + consensus window live in ModelData.Threshold / .ConsensusS
        // (embedded from _model.json). Mirrored here only for the settle-latency math.
        private const int    CONSENSUS_SETTLE_SEC = 180;        // = ModelData.ConsensusS

        // ---- z_se native-derivation constants (core_v2 statistical_field_engine) ----
        private const int    ZSE_WINDOW_1M = 15;                // N_BASE[1m]=15 -> L3_1m_z_se_15
        private const int    ZSE_OLS_DDOF  = 2;                 // residual std ddof=2 (endpoint OLS)

        // ---- session calendar (America/Chicago RTH; NT8 exchange-local) ----
        private const int RTH_OPEN_HH = 8,  RTH_OPEN_MM = 30;   // 08:30 CT
        private const int RTH_CLOSE_HH = 15, RTH_CLOSE_MM = 15; // 15:15 CT (bar-close gate)
        private const int B9_HH = 8, B9_MM = 35;                // opening-range = 08:30-08:35 CT

        // ---- secondary-series indices ----
        private const int BIP_5S = 0;   // primary chart series MUST be 5s
        private const int BIP_1M = 1;   // native z_se OLS lives here (P2-3)

        // ---- embedded engine (shared core) ----
        private Tmpl0 tmpl;             // frozen codebook (constants; no file IO)

        // ---- per-session 5s buffers (the batch-core input, streamed) ----
        private List<long>   bTs = new List<long>();
        private List<double> bO = new List<double>(), bH = new List<double>(),
                             bL = new List<double>(), bC = new List<double>(), bV = new List<double>();
        private List<bool>   bRth = new List<bool>(), bB9 = new List<bool>();
        private List<double> bTod = new List<double>();
        private List<double> bZse = new List<double>();
        private List<PriorDay> prior = new List<PriorDay>();

        // native 1m z_se state (P2-3)
        private List<double> min1Close = new List<double>();
        private double lastZse = double.NaN;   // ffilled onto 5s rows

        // prior-day rolling extremes (P2-12: exact prior-daily equivalence pending)
        private double pdHi = double.NegativeInfinity, pdLo = double.PositiveInfinity, pdClose = double.NaN;
        private bool   pdHasBars = false;

        // ---- streaming decision state ----
        private long   prevBarMin = long.MinValue;      // detects 1m completion
        private DateTime sessionDate = DateTime.MinValue;
        private bool   tradeAllowedToday = true;
        private int    openDir = 0;                     // +1 long / -1 short / 0 flat
        private double openEntryPrice = double.NaN;
        private int    dayStartIdx = 0;                 // Ctx.Start (P2-12 warmup/tail)

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Ensemble entry combiner (22-stream logistic P >= 0.713983) with " +
                    "R-trigger ride-only reversal exit. Architecture B, mechanical manager, no cut " +
                    "logic. Decision core = bit-exact port of research/nt8_port (100% fire/entry/pivot " +
                    "parity, 20 reference days). RC -- NOT deploy-approved.";
                Name = "EnsembleRunner_v0.2-RC";
                Calculate = Calculate.OnBarClose;          // closed-bar semantics (no lookahead)
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 60;
                BarsRequiredToTrade = 1;
                IsInstantiatedOnEachOptimizationIteration = true;

                Quantity = 1;
                EnableCatastrophicStop = false;            // OFF in SIM by default
                CatastrophicStopPoints = 200;              // present for live; ignored while OFF
                SessionFlattenHH = 15;                     // 15:55 CT flatten guard
                SessionFlattenMM = 55;
                WarmupTailBars5s = 2500;                   // prior-day tail (P2-12); ~ harness Start
                ZSeNative = true;                          // native OLS z_se (P2-3)
            }
            else if (State == State.Configure)
            {
                // Primary series (index 0) MUST be 5-second bars (the harness substrate).
                // TODO(P2-1b): assert BarsPeriod == 5s at DataLoaded; reject otherwise.
                AddDataSeries(BarsPeriodType.Minute, 1);   // BIP_1M = 1 : native z_se OLS
            }
            else if (State == State.DataLoaded)
            {
                tmpl = new Tmpl0();                         // embedded codebook (P2-2 / P2-4)
            }
        }

        protected override void OnBarUpdate()
        {
            // ---------- 1-minute series: native z_se (P2-3) ----------
            if (BarsInProgress == BIP_1M)
            {
                if (ZSeNative && CurrentBars[BIP_1M] >= ZSE_WINDOW_1M - 1)
                {
                    // Endpoint OLS z on the last 15 1m closes, residual std ddof=2 -- same
                    // formula family as core_v2 _ols_fit_kernel / harness MathX.Z21 (window 21).
                    // TODO(P2-3): bit-parity vs core_v2 statistical_field_engine before live.
                    lastZse = OlsEndpointZ(BIP_1M, ZSE_WINDOW_1M, ZSE_OLS_DDOF);
                }
                return;
            }

            // ---------- 5-second primary series: buffer + decide ----------
            if (BarsInProgress != BIP_5S) return;

            DateTime t = Times[BIP_5S][0];
            RollSession(t);                                // new-day reset of buffers + guards

            long ts = ToEpoch(t);
            long barMin = (ts / 60) * 60;

            // append this closed 5s bar to the day buffer (the batch-core input)
            bTs.Add(ts);
            bO.Add(Opens[BIP_5S][0]); bH.Add(Highs[BIP_5S][0]); bL.Add(Lows[BIP_5S][0]);
            bC.Add(Closes[BIP_5S][0]); bV.Add(Volumes[BIP_5S][0]);
            bool inRth = IsRth(t);                          // TODO(P2-5): DST-correct CT session gate
            bRth.Add(inRth);
            bB9.Add(IsBefore9(t));                          // opening-range window (P2-5)
            bTod.Add(TimeOfDayFrac(t));
            bZse.Add(lastZse);                              // ffilled 1m z_se (P2-3)

            // detect completion of the PREVIOUS 1-minute bar
            if (prevBarMin != long.MinValue && barMin != prevBarMin)
                RunDecision(prevBarMin, t);
            prevBarMin = barMin;

            // ---- catastrophic stop (live only; OFF in SIM by default) ----
            if (EnableCatastrophicStop && openDir != 0)
            {
                double adverse = openDir > 0
                    ? (openEntryPrice - Lows[BIP_5S][0])
                    : (Highs[BIP_5S][0] - openEntryPrice);
                if (adverse >= CatastrophicStopPoints) FlattenPosition("CatastrophicStop");
                // TODO(P2-10): real ExitLongStopMarket for live, not an intrabar poll.
            }

            // ---- session flatten guard (15:55 CT): flatten + block new entries ----
            if (AtOrAfter(t, SessionFlattenHH, SessionFlattenMM))
            {
                if (openDir != 0) FlattenPosition("SessionFlatten");
                tradeAllowedToday = false;                  // P2-11 (DST-correct time = P2-5)
            }
        }

        // Re-run the proven batch core over the day-so-far; act on the completed minute.
        private void RunDecision(long curMin, DateTime now)
        {
            int n = bTs.Count;
            if (n < 2) return;

            Ctx x = new Ctx();
            x.Day = sessionDate.ToString("yyyy_MM_dd");
            x.N = n;
            x.Start = Math.Min(dayStartIdx, n - 1);         // Ctx.Start (P2-12 warmup/tail)
            x.Ts = bTs.ToArray();
            x.O = bO.ToArray(); x.H = bH.ToArray(); x.L = bL.ToArray();
            x.C = bC.ToArray(); x.V = bV.ToArray();
            x.Rth = bRth.ToArray(); x.Before9 = bB9.ToArray(); x.Tod = bTod.ToArray();
            x.Zse = bZse.ToArray();
            bool hasZ = false;
            for (int i = 0; i < n; i++) if (Pd.Fin(x.Zse[i])) { hasZ = true; break; }
            x.HasZse = hasZ;
            x.Prior = prior;
            x.BuildDayCtx();

            // TODO(P2-perf): ProcessDay is O(N) per minute (~390 calls/day). Fine for
            // backtest/live; an incremental per-generator port would cut it to O(1)/bar.
            List<BarRec> recs = V02Core.ProcessDay(x, tmpl);
            Dictionary<long, BarRec> byTs = new Dictionary<long, BarRec>();
            for (int k = 0; k < recs.Count; k++) byTs[recs[k].BarTs] = recs[k];

            // ---- EXIT first: R-trigger reversal against the open leg (ride-only) ----
            // zz_confirm is CAUSAL -> final at minute close (no settle needed).
            BarRec cur;
            if (openDir != 0 && byTs.TryGetValue(curMin, out cur)
                && cur.ZzConfirm != 0 && cur.ZzConfirm == -openDir)
            {
                FlattenPosition("RTriggerReversal");        // P2-9
            }

            // ---- ENTRY: act on the minute whose ±180s consensus has now settled ----
            long settled = curMin - CONSENSUS_SETTLE_SEC;
            BarRec s;
            if (openDir == 0 && tradeAllowedToday && IsRth(now)
                && byTs.TryGetValue(settled, out s) && s.Entry == 1 && s.EntryDir != 0)
            {
                // TODO(P2-8): fill semantics -- this acts ~180s after the signal minute
                //   (the consensus settle). EntriesPerDirection=1 caps to one open leg.
                if (s.EntryDir > 0) EnterLong(Quantity, "Long");
                else                EnterShort(Quantity, "Short");
            }
        }

        protected override void OnPositionUpdate(Position position, double averagePrice,
                                                 int quantity, MarketPosition marketPosition)
        {
            if (marketPosition == MarketPosition.Flat) { openDir = 0; openEntryPrice = double.NaN; }
            else
            {
                openDir = marketPosition == MarketPosition.Long ? 1 : -1;
                openEntryPrice = averagePrice;
            }
        }

        // ------------------------------------------------------------------ helpers
        private void FlattenPosition(string reason)
        {
            if (Position.MarketPosition == MarketPosition.Long)       ExitLong("X_" + reason, "Long");
            else if (Position.MarketPosition == MarketPosition.Short) ExitShort("X_" + reason, "Short");
            openDir = 0;
        }

        private void RollSession(DateTime t)
        {
            if (t.Date != sessionDate)
            {
                // roll prior-day extremes into the Prior list (P2-12: exact prior-daily
                // profile equivalence vs the harness export is still pending).
                if (pdHasBars)
                {
                    PriorDay pd = new PriorDay();
                    pd.High = pdHi; pd.Low = pdLo; pd.Close = pdClose;
                    prior.Add(pd);
                }
                sessionDate = t.Date;
                tradeAllowedToday = true;
                prevBarMin = long.MinValue;
                lastZse = double.NaN;
                min1Close.Clear();
                pdHi = double.NegativeInfinity; pdLo = double.PositiveInfinity; pdClose = double.NaN;
                pdHasBars = false;
                // NOTE(P2-12): the harness streams a prior-day 5s TAIL (Start~2500 rows) so the
                //   R-trigger + rolling windows are warm at RTH open. Here the buffer resets per
                //   session; WarmupTailBars5s approximates Start. Exact tail equivalence = P2-12.
                bTs.Clear(); bO.Clear(); bH.Clear(); bL.Clear(); bC.Clear(); bV.Clear();
                bRth.Clear(); bB9.Clear(); bTod.Clear(); bZse.Clear();
                dayStartIdx = 0;
            }
            // track this bar's contribution to the (developing) prior-day extremes
            double hi = Highs[BIP_5S][0], lo = Lows[BIP_5S][0], cl = Closes[BIP_5S][0];
            if (hi > pdHi) pdHi = hi;
            if (lo < pdLo) pdLo = lo;
            pdClose = cl; pdHasBars = true;
        }

        private static long ToEpoch(DateTime t)
        {
            // NT8 bar times are exchange-local DateTimeKind.Unspecified; treat as UTC-agnostic
            // epoch seconds for bucketing (ts//period). Absolute offset is irrelevant to the
            // floor-division buckets the core uses. // TODO(P2-5): confirm vs harness ts basis.
            return (long)(t - new DateTime(1970, 1, 1)).TotalSeconds;
        }

        private bool IsRth(DateTime t)
        {
            int mins = t.Hour * 60 + t.Minute;
            return mins >= (RTH_OPEN_HH * 60 + RTH_OPEN_MM) &&
                   mins <= (RTH_CLOSE_HH * 60 + RTH_CLOSE_MM);
        }
        private bool IsBefore9(DateTime t)
        {
            int mins = t.Hour * 60 + t.Minute;
            return mins >= (RTH_OPEN_HH * 60 + RTH_OPEN_MM) && mins < (B9_HH * 60 + B9_MM);
        }
        private double TimeOfDayFrac(DateTime t)
        {
            int open = RTH_OPEN_HH * 60 + RTH_OPEN_MM, close = RTH_CLOSE_HH * 60 + RTH_CLOSE_MM;
            int mins = t.Hour * 60 + t.Minute;
            double f = (mins - open) / (double)(close - open);
            return f < 0 ? 0 : (f > 1 ? 1 : f);            // TODO(P2-5): match harness tod exactly
        }
        private bool AtOrAfter(DateTime t, int hh, int mm)
        {
            return (t.Hour * 60 + t.Minute) >= (hh * 60 + mm);
        }

        // Endpoint OLS z on the BIP_1M close series. Mirrors MathX.Z21 (window 21) with
        // window 15. // TODO(P2-3): bit-parity vs core_v2 _ols_fit_kernel before live.
        private double OlsEndpointZ(int bip, int window, int ddof)
        {
            if (CurrentBars[bip] < window - 1) return double.NaN;
            double xm = (window - 1) / 2.0, xv = 0, ym = 0;
            for (int k = 0; k < window; k++) { double dx = k - xm; xv += dx * dx; ym += Closes[bip][k]; }
            ym /= window;
            double num = 0;
            for (int k = 0; k < window; k++) num += (Closes[bip][window - 1 - k] - ym) * (k - xm);
            double slope = num / xv, inter = ym - slope * xm;
            double ss = 0;
            for (int k = 0; k < window; k++)
            {
                double fit = slope * k + inter;
                double r = Closes[bip][window - 1 - k] - fit;
                ss += r * r;
            }
            double var = ss / (window - ddof);
            double sd = Math.Sqrt(Math.Max(var, 0));
            double fitLast = slope * (window - 1) + inter;
            return sd > 1e-10 ? (Closes[bip][0] - fitLast) / sd : double.NaN;
        }

        #region Properties
        [NinjaScriptProperty, Range(1, int.MaxValue)]
        [Display(Name = "Quantity", Order = 1, GroupName = "1. Sizing")]
        public int Quantity { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable catastrophic stop (LIVE)", Order = 1, GroupName = "2. Risk")]
        public bool EnableCatastrophicStop { get; set; }

        [NinjaScriptProperty, Range(1, double.MaxValue)]
        [Display(Name = "Catastrophic stop (points)", Order = 2, GroupName = "2. Risk")]
        public double CatastrophicStopPoints { get; set; }

        [NinjaScriptProperty, Range(0, 23)]
        [Display(Name = "Session flatten hour (CT)", Order = 1, GroupName = "3. Session")]
        public int SessionFlattenHH { get; set; }

        [NinjaScriptProperty, Range(0, 59)]
        [Display(Name = "Session flatten minute (CT)", Order = 2, GroupName = "3. Session")]
        public int SessionFlattenMM { get; set; }

        [NinjaScriptProperty, Range(0, int.MaxValue)]
        [Display(Name = "Warmup tail bars (5s)", Order = 1, GroupName = "4. Warmup")]
        public int WarmupTailBars5s { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Native z_se (OLS)", Order = 1, GroupName = "5. Features")]
        public bool ZSeNative { get; set; }
        #endregion
    }

    // =========================================================================
    // SHARED DECISION CORE -- injected VERBATIM from the single source of truth
    //   research/nt8_port/csharp/v02/EnsembleCoreV02.region.cs
    // by v02/assemble.py; re-checked byte-for-byte by v02/verify_region.py; proven
    // 100.000% vs golden by v02/shim (V02ParityShim). DO NOT hand-edit below.
    // =========================================================================
    namespace EnsembleV02Core
    {
// ===SHARED-CORE-V02 BEGIN=== (single source: research/nt8_port/csharp/v02/EnsembleCoreV02.region.cs)
using System;
using System.Collections.Generic;

    // ===================================================================
    // ENSEMBLE CORE v0.2 -- decision core, C# 7.3-clean, ValueTuple-free.
    // Down-levelled VERBATIM port of the parity harness
    //   research/nt8_port/csharp/{Pandas,Model,Gens,Tmpl0,Program}.cs
    // (100.000% bit-parity vs Python golden, 178,640/178,640 cells, 20 days).
    // Deviations from the harness are ONLY those forced by C#7.3 / .NET4.8:
    //   (D1) named ValueTuples -> plain structs (NmpFire/Nmp9Ev/TmplSub/ZzResult/DmiAdx);
    //   (D2) Math.Log2(x) -> Log2(x) = Math.Log(x)/Math.Log(2.0);
    //   (D3) JSON codebook load -> embedded ModelData/Tmpl0Data constants;
    //   (D4) LINQ OrderBy/ThenBy stable sort -> explicit index sort (Ts, then index);
    //   (D5) TMPL0 diagnostic Debug capture dropped (write-only in harness).
    // Each is arithmetic-neutral; proven by V02ParityShim (byte-identical output).
    // ===================================================================

    struct Fire
    {
        public int Row; public long Ts; public bool IsLong; public double Value;
        public double PivotAgeMin; public int SigWithLeg; public double Tod;
        public string Det; public long Tf;
    }

    struct PriorDay { public double High, Low, Close; }

    // (D1) tuple replacements
    struct NmpFire { public int I; public double Z; }
    struct Nmp9Ev { public int I; public bool IsLong; public string Tier; public double Val; }
    struct TmplSub { public long Tf; public double Conv; public int Dir; }
    struct ZzResult { public int[] Dir; public int[] Flip; public int[] PivBar; public double[] PivPx; public int MinRev; }

    // ---- pandas/numpy-exact helpers (Pandas.cs, verbatim) -----------------
    static class Pd
    {
        public static bool Fin(double x) { return !double.IsNaN(x) && !double.IsInfinity(x); }

        public static double[] EwmAlpha(double[] x, double alpha)
        {
            int n = x.Length; var y = new double[n];
            double oldWtFactor = 1.0 - alpha, newWt = alpha;
            double weighted = double.NaN, oldWt = 1.0; bool seeded = false;
            for (int i = 0; i < n; i++)
            {
                double cur = x[i]; bool isObs = Fin(cur);
                if (!seeded)
                {
                    if (isObs) { weighted = cur; oldWt = 1.0; seeded = true; }
                    y[i] = seeded ? weighted : double.NaN;
                    continue;
                }
                oldWt *= oldWtFactor;
                if (isObs)
                {
                    if (weighted != cur)
                        weighted = (oldWt * weighted + newWt * cur) / (oldWt + newWt);
                    oldWt = 1.0;
                }
                y[i] = weighted;
            }
            return y;
        }
        public static double[] EwmSpan(double[] x, double span) { return EwmAlpha(x, 2.0 / (span + 1.0)); }
        public static double[] EwmCom(double[] x, double com) { return EwmAlpha(x, 1.0 / (1.0 + com)); }

        public static double[] RollMean(double[] x, int w, int minp)
        {
            int n = x.Length; var y = new double[n];
            for (int i = 0; i < n; i++)
            {
                if (i < w - 1) { y[i] = double.NaN; continue; }
                double s = 0; int cnt = 0;
                for (int j = i - w + 1; j <= i; j++) if (Fin(x[j])) { s += x[j]; cnt++; }
                y[i] = cnt >= minp ? s / cnt : double.NaN;
            }
            return y;
        }
        public static double[] RollStd(double[] x, int w, int minp, int ddof)
        {
            int n = x.Length; var y = new double[n];
            for (int i = 0; i < n; i++)
            {
                if (i < w - 1) { y[i] = double.NaN; continue; }
                double s = 0; int cnt = 0;
                for (int j = i - w + 1; j <= i; j++) if (Fin(x[j])) { s += x[j]; cnt++; }
                if (cnt < minp || cnt - ddof <= 0) { y[i] = double.NaN; continue; }
                double m = s / cnt, ss = 0;
                for (int j = i - w + 1; j <= i; j++) if (Fin(x[j])) { double d = x[j] - m; ss += d * d; }
                y[i] = Math.Sqrt(ss / (cnt - ddof));
            }
            return y;
        }
        public static double[] RollMax(double[] x, int w, int minp)
        {
            int n = x.Length; var y = new double[n];
            for (int i = 0; i < n; i++)
            {
                if (i < w - 1) { y[i] = double.NaN; continue; }
                double mx = double.NegativeInfinity; int cnt = 0;
                for (int j = i - w + 1; j <= i; j++) if (Fin(x[j])) { if (x[j] > mx) mx = x[j]; cnt++; }
                y[i] = cnt >= minp ? mx : double.NaN;
            }
            return y;
        }
        public static double[] RollMin(double[] x, int w, int minp)
        {
            int n = x.Length; var y = new double[n];
            for (int i = 0; i < n; i++)
            {
                if (i < w - 1) { y[i] = double.NaN; continue; }
                double mn = double.PositiveInfinity; int cnt = 0;
                for (int j = i - w + 1; j <= i; j++) if (Fin(x[j])) { if (x[j] < mn) mn = x[j]; cnt++; }
                y[i] = cnt >= minp ? mn : double.NaN;
            }
            return y;
        }
        public static double[] RollSum(double[] x, int w, int minp)
        {
            int n = x.Length; var y = new double[n];
            for (int i = 0; i < n; i++)
            {
                if (i < w - 1) { y[i] = double.NaN; continue; }
                double s = 0; int cnt = 0;
                for (int j = i - w + 1; j <= i; j++) if (Fin(x[j])) { s += x[j]; cnt++; }
                y[i] = cnt >= minp ? s : double.NaN;
            }
            return y;
        }
        public static double[] Diff(double[] x)
        {
            int n = x.Length; var y = new double[n]; y[0] = double.NaN;
            for (int i = 1; i < n; i++) y[i] = x[i] - x[i - 1];
            return y;
        }
        public static double[] Shift(double[] x, int k)
        {
            int n = x.Length; var y = new double[n];
            for (int i = 0; i < n; i++) y[i] = (i - k >= 0 && i - k < n) ? x[i - k] : double.NaN;
            return y;
        }
        // (D2) Math.Log2 shim -- .NET4.8 lacks Math.Log2; proven parity-neutral by shim.
        public static double Log2(double x) { return Math.Log(x) / Math.Log(2.0); }
    }

    // Clock-aligned OHLCV buckets (Pandas.cs Buckets, verbatim).
    class Buckets
    {
        public long[] Ids;
        public double[] O, H, L, C, V;
        public int[] CloseRow;
        public int[] RowClosed;
        public Dictionary<long, int> Pos;

        public static Buckets Build(long[] ts, double[] o, double[] h, double[] l, double[] c, double[] v, long period)
        {
            int n = ts.Length;
            var b = new Buckets();
            var ids = new List<long>();
            var idsO = new List<double>(); var idsH = new List<double>(); var idsL = new List<double>();
            var idsC = new List<double>(); var idsV = new List<double>();
            var firstRow = new List<int>();
            b.Pos = new Dictionary<long, int>();
            long cur = long.MinValue; int p = -1;
            for (int i = 0; i < n; i++)
            {
                long bid = ts[i] / period;
                if (bid != cur)
                {
                    cur = bid; p++;
                    ids.Add(bid); b.Pos[bid] = p;
                    idsO.Add(o[i]); idsH.Add(h[i]); idsL.Add(l[i]); idsC.Add(c[i]); idsV.Add(v[i]);
                    firstRow.Add(i);
                }
                else
                {
                    if (h[i] > idsH[p]) idsH[p] = h[i];
                    if (l[i] < idsL[p]) idsL[p] = l[i];
                    idsC[p] = c[i]; idsV[p] += v[i];
                }
            }
            int m = ids.Count;
            b.Ids = ids.ToArray(); b.O = idsO.ToArray(); b.H = idsH.ToArray();
            b.L = idsL.ToArray(); b.C = idsC.ToArray(); b.V = idsV.ToArray();
            b.CloseRow = new int[m];
            for (int k = 0; k < m; k++)
            {
                long wantNext = b.Ids[k] + 1;
                int np2;
                if (b.Pos.TryGetValue(wantNext, out np2)) b.CloseRow[k] = firstRow[np2];
                else b.CloseRow[k] = -1;
            }
            b.RowClosed = new int[n];
            for (int i = 0; i < n; i++)
            {
                long want = ts[i] / period - 1;
                int pp;
                b.RowClosed[i] = b.Pos.TryGetValue(want, out pp) ? pp : -1;
            }
            return b;
        }
    }

    // DayCtx equivalent (Model.cs Ctx, verbatim).
    class Ctx
    {
        public string Day;
        public int Start, N;
        public long[] Ts;
        public double[] O, H, L, C, V;
        public bool[] Rth, Before9;
        public double[] Tod;
        public double[] Zse;
        public bool HasZse;
        public List<PriorDay> Prior = new List<PriorDay>();

        public const double TICK = 0.25;
        public const int BAR_1M = 12;
        public const int ATR_N = 14;
        public const double ATR_MULT = 4.0;

        public double[] ZzThr;
        public int[] PivI;
        public sbyte[] Leg;
        public sbyte[] PivConfirm;

        public void BuildDayCtx()
        {
            BuildZzThr();
            BuildZigzag();
        }

        void BuildZzThr()
        {
            int nb = (N + BAR_1M - 1) / BAR_1M;
            var c1 = new double[nb]; var h1 = new double[nb]; var l1 = new double[nb];
            for (int b = 0; b < nb; b++)
            {
                int s = b * BAR_1M, e = Math.Min(s + BAR_1M, N);
                double hi = double.NegativeInfinity, lo = double.PositiveInfinity;
                for (int i = s; i < e; i++) { if (H[i] > hi) hi = H[i]; if (L[i] < lo) lo = L[i]; }
                h1[b] = hi; l1[b] = lo; c1[b] = C[e - 1];
            }
            var tr1 = new double[nb];
            for (int b = 0; b < nb; b++)
            {
                double pc = b > 0 ? c1[b - 1] : double.NaN;
                double a = h1[b] - l1[b];
                double mx = a;
                if (Pd.Fin(pc)) { mx = Math.Max(mx, Math.Abs(h1[b] - pc)); mx = Math.Max(mx, Math.Abs(l1[b] - pc)); }
                tr1[b] = mx;
            }
            var atr1 = Pd.RollMean(tr1, ATR_N, ATR_N);
            ZzThr = new double[N];
            for (int i = 0; i < N; i++) ZzThr[i] = atr1[i / BAR_1M] * ATR_MULT;
        }

        void BuildZigzag()
        {
            PivI = new int[N]; Leg = new sbyte[N]; PivConfirm = new sbyte[N];
            int hi_i = 0, lo_i = 0, d = 0, last = 0;
            double hi_v = C[0], lo_v = C[0];
            for (int i = 1; i < N; i++)
            {
                double x = C[i];
                double t = Pd.Fin(ZzThr[i]) ? ZzThr[i] : double.PositiveInfinity;
                if (x > hi_v) { hi_v = x; hi_i = i; }
                if (x < lo_v) { lo_v = x; lo_i = i; }
                if (d >= 0 && hi_v - x >= t) { last = hi_i; d = -1; lo_v = x; lo_i = i; PivConfirm[i] = -1; }
                else if (d <= 0 && x - lo_v >= t) { last = lo_i; d = 1; hi_v = x; hi_i = i; PivConfirm[i] = 1; }
                PivI[i] = last; Leg[i] = (sbyte)d;
            }
        }

        public Fire Emit(int i, bool isLong, double value, string det)
        {
            int swl = Leg[i] != 0 ? ((Leg[i] > 0) == isLong ? 1 : 0) : 0;
            var f = new Fire();
            f.Row = i; f.Ts = Ts[i]; f.IsLong = isLong; f.Value = value;
            f.PivotAgeMin = (i - PivI[i]) * 5.0 / 60.0;
            f.SigWithLeg = swl; f.Tod = Tod[i]; f.Det = det; f.Tf = 0;
            return f;
        }

        public IEnumerable<int> RthIdx()
        {
            for (int i = 0; i < N; i++) if (Rth[i] && i >= Start) yield return i;
        }
    }

    static class MathX
    {
        public static double[] Z21(double[] c)
        {
            int n = c.Length, w = 21; var z = new double[n];
            for (int i = 0; i < n; i++) z[i] = double.NaN;
            if (n < w) return z;
            double xm = (w - 1) / 2.0, xv = 0;
            for (int k = 0; k < w; k++) { double dx = k - xm; xv += dx * dx; }
            for (int i = w - 1; i < n; i++)
            {
                int s = i - w + 1;
                double ym = 0; for (int k = 0; k < w; k++) ym += c[s + k]; ym /= w;
                double num = 0; for (int k = 0; k < w; k++) num += (c[s + k] - ym) * (k - xm);
                double slope = num / xv, inter = ym - slope * xm;
                double ss = 0;
                for (int k = 0; k < w; k++) { double fit = slope * k + inter; double r = c[s + k] - fit; ss += r * r; }
                double var = ss / (w - 2); double sd = Math.Sqrt(Math.Max(var, 0));
                double fitLast = slope * (w - 1) + inter;
                z[i] = sd > 0 ? (c[i] - fitLast) / sd : double.NaN;
            }
            return z;
        }

        public static double[] WilderDmiDiff(double[] h, double[] l, double[] c)
        {
            int n = h.Length;
            var up = Pd.Diff(h); var dnRaw = Pd.Diff(l);
            var dmp = new double[n]; var dmm = new double[n];
            for (int i = 0; i < n; i++)
            {
                double u = up[i]; double dn = -dnRaw[i];
                dmp[i] = (Pd.Fin(u) && Pd.Fin(dn) && u > dn && u > 0) ? u : 0.0;
                dmm[i] = (Pd.Fin(u) && Pd.Fin(dn) && dn > u && dn > 0) ? dn : 0.0;
            }
            var pc = Pd.Shift(c, 1);
            var tr = new double[n];
            for (int i = 0; i < n; i++)
            {
                double a = h[i] - l[i]; double mx = a;
                if (Pd.Fin(pc[i])) { mx = Math.Max(mx, Math.Abs(h[i] - pc[i])); mx = Math.Max(mx, Math.Abs(l[i] - pc[i])); }
                tr[i] = mx;
            }
            var trs = Pd.EwmAlpha(tr, 1.0 / 14.0);
            var dips = Pd.EwmAlpha(dmp, 1.0 / 14.0);
            var dims = Pd.EwmAlpha(dmm, 1.0 / 14.0);
            var dmi = new double[n];
            for (int i = 0; i < n; i++)
            {
                double trv = trs[i];
                double dip = (Pd.Fin(trv) && trv != 0) ? 100.0 * dips[i] / trv : double.NaN;
                double dim = (Pd.Fin(trv) && trv != 0) ? 100.0 * dims[i] / trv : double.NaN;
                dmi[i] = dip - dim;
            }
            return dmi;
        }
    }

    class TfState
    {
        public Buckets B;
        public double[] Z, Vel, Acc, Wick, Vr, Volr, Dmi;
        public int[] RowClosed;

        public static TfState Build(Ctx ctx, long period)
        {
            var b = Buckets.Build(ctx.Ts, ctx.O, ctx.H, ctx.L, ctx.C, ctx.V, period);
            int m = b.Ids.Length;
            var t = new TfState(); t.B = b; t.RowClosed = b.RowClosed;
            t.Z = MathX.Z21(b.C);
            t.Vel = new double[m]; t.Acc = new double[m]; t.Wick = new double[m];
            var vel = t.Vel;
            for (int i = 0; i < m; i++) vel[i] = i == 0 ? double.NaN : (b.C[i] - b.C[i - 1]) / Ctx.TICK;
            for (int i = 0; i < m; i++) t.Acc[i] = i == 0 ? double.NaN : (Pd.Fin(vel[i]) && Pd.Fin(vel[i - 1]) ? vel[i] - vel[i - 1] : double.NaN);
            for (int i = 0; i < m; i++)
            {
                double rng = Math.Max(b.H[i] - b.L[i], 1e-9);
                t.Wick[i] = 1.0 - Math.Abs(b.C[i] - b.O[i]) / rng;
            }
            var s10 = Pd.RollStd(b.C, 10, 10, 1);
            var s60 = Pd.RollStd(b.C, 60, 60, 1);
            t.Vr = new double[m];
            for (int i = 0; i < m; i++)
            {
                double a = s60[i];
                t.Vr[i] = (Pd.Fin(a) && a != 0) ? s10[i] / a : double.NaN;
            }
            var vmean = Pd.RollMean(b.V, 30, 30);
            t.Volr = new double[m];
            for (int i = 0; i < m; i++) t.Volr[i] = b.V[i] / vmean[i];
            t.Dmi = MathX.WilderDmiDiff(b.H, b.L, b.C);
            return t;
        }
        public int At(int row) { return RowClosed[row]; }
    }

    // ---- 21 generators (Gens.cs, verbatim; tuples -> structs) --------------
    static class Gens
    {
        const int COOLDOWN = 60;

        public static List<Fire> Zigzag(Ctx x)
        {
            var o = new List<Fire>();
            foreach (int i in x.RthIdx())
                if (x.PivConfirm[i] != 0)
                    o.Add(x.Emit(i, x.PivConfirm[i] > 0, Pd.Fin(x.ZzThr[i]) ? x.ZzThr[i] : 0.0, "ZIGZAG"));
            return o;
        }

        public static List<Fire> Orb02(Ctx x)
        {
            var idx = new List<int>(); foreach (int i in x.RthIdx()) idx.Add(i);
            double orh = double.NegativeInfinity, orl = double.PositiveInfinity; bool any = false;
            foreach (int i in idx) if (x.Before9[i]) { any = true; if (x.C[i] > orh) orh = x.C[i]; if (x.C[i] < orl) orl = x.C[i]; }
            if (!any) return new List<Fire>();
            foreach (int i in idx)
            {
                if (x.Before9[i]) continue;
                if (x.C[i] > orh) return new List<Fire> { x.Emit(i, true, x.C[i] - orh, "ORB02") };
                if (x.C[i] < orl) return new List<Fire> { x.Emit(i, false, orl - x.C[i], "ORB02") };
            }
            return new List<Fire>();
        }

        public static List<Fire> Round05(Ctx x)
        {
            const double GRID = 50.0, PRIME = 5.0;
            var o = new List<Fire>();
            var primB = new Dictionary<double, bool>(); var primS = new Dictionary<double, bool>();
            for (int i = 0; i < x.N; i++)
            {
                double p = x.C[i];
                double base_ = (double)((long)(p / GRID)) * GRID;
                double[] levels = { base_ - GRID, base_, base_ + GRID };
                foreach (double L in levels)
                {
                    bool bb;
                    if (p >= L && primB.TryGetValue(L, out bb) && bb)
                    { primB[L] = false; if (x.Rth[i] && i >= x.Start) o.Add(x.Emit(i, true, PRIME, "ROUND05")); }
                    bool ss;
                    if (p <= L && primS.TryGetValue(L, out ss) && ss)
                    { primS[L] = false; if (x.Rth[i] && i >= x.Start) o.Add(x.Emit(i, false, PRIME, "ROUND05")); }
                    if (p < L - PRIME) primB[L] = true; else if (p >= L) primB[L] = false;
                    if (p > L + PRIME) primS[L] = true; else if (p <= L) primS[L] = false;
                }
            }
            return o;
        }

        public static List<Fire> Vwap03(Ctx x)
        {
            var o = new List<Fire>();
            double cumPv = 0, cumVol = 0; var buf = new List<double>(); bool pb = false, pbear = false; double zprev = 0;
            for (int i = 0; i < x.N; i++)
            {
                if (!x.Rth[i]) { cumPv = 0; cumVol = 0; buf.Clear(); pb = false; pbear = false; zprev = 0; continue; }
                cumPv += x.C[i] * x.V[i]; cumVol += x.V[i];
                double vwap = cumVol == 0 ? x.C[i] : cumPv / cumVol;
                buf.Add(x.C[i]); if (buf.Count > 20) buf.RemoveAt(0);
                if (buf.Count < 20) continue;
                double sd = Std(buf, 1);
                double z = (x.C[i] - vwap) / Math.Max(0.25, sd);
                bool fireBear = pbear && z < zprev && z > 0;
                bool fireBull = pb && z > zprev && z < 0;
                if (z > 2.0 && zprev <= 2.0) pbear = true; else if (fireBear || z <= 0) pbear = false;
                if (z < -2.0 && zprev >= -2.0) pb = true; else if (fireBull || z >= 0) pb = false;
                if (i >= x.Start)
                {
                    if (fireBear) o.Add(x.Emit(i, false, z, "VWAP03"));
                    if (fireBull) o.Add(x.Emit(i, true, -z, "VWAP03"));
                }
                zprev = z;
            }
            return o;
        }
        static double Std(List<double> b, int ddof)
        {
            int n = b.Count; if (n - ddof <= 0) return double.NaN;
            double m = 0; foreach (var v in b) m += v; m /= n;
            double s = 0; foreach (var v in b) { double d = v - m; s += d * d; }
            return Math.Sqrt(s / (n - ddof));
        }

        public static List<Fire> Dow19(Ctx x)
        {
            var vs = Pd.RollMean(x.V, 20, 20);
            var cs1 = Pd.Shift(x.C, 1);
            var hi10 = Pd.RollMax(cs1, 10, 10);
            var lo10 = Pd.RollMin(cs1, 10, 10);
            var o = new List<Fire>(); int cool = 0;
            foreach (int i in x.RthIdx())
            {
                if (i < 21 || !Pd.Fin(vs[i])) continue;
                if (cool > 0) { cool--; continue; }
                if (x.V[i] < vs[i])
                {
                    if (x.C[i] > hi10[i]) { o.Add(x.Emit(i, false, x.C[i] - hi10[i], "DOW19")); cool = COOLDOWN; }
                    else if (x.C[i] < lo10[i]) { o.Add(x.Emit(i, true, lo10[i] - x.C[i], "DOW19")); cool = COOLDOWN; }
                }
            }
            return o;
        }

        public static List<Fire> Tunnel20(Ctx x)
        {
            var eh = Pd.EwmSpan(x.H, 34); var el = Pd.EwmSpan(x.L, 34);
            var o = new List<Fire>(); int cool = 0;
            foreach (int i in x.RthIdx())
            {
                if (i < 1) continue;
                if (cool > 0) { cool--; continue; }
                if (x.C[i - 1] <= eh[i - 1] && x.C[i] > eh[i]) { o.Add(x.Emit(i, true, x.C[i] - eh[i], "TUNNEL20")); cool = COOLDOWN; }
                else if (x.C[i - 1] >= el[i - 1] && x.C[i] < el[i]) { o.Add(x.Emit(i, false, el[i] - x.C[i], "TUNNEL20")); cool = COOLDOWN; }
            }
            return o;
        }

        public static List<Fire> Atr09(Ctx x)
        {
            var pd = x.Prior; int np = pd.Count;
            if (np < 15) return new List<Fire>();
            double atr = 0;
            for (int j = np - 14; j < np; j++)
            {
                var dj = pd[j]; var dp = pd[j - 1];
                double tr = Math.Max(dj.High - dj.Low, Math.Max(Math.Abs(dj.High - dp.Close), Math.Abs(dj.Low - dp.Close)));
                atr += tr;
            }
            atr /= 14.0;
            var o = new List<Fire>();
            double rh = double.NegativeInfinity, rl = double.PositiveInfinity;
            double[] xs = { 0.5, 0.75, 1.0 }; var trig = new bool[3];
            foreach (int i in x.RthIdx())
            {
                double p = x.C[i]; rh = Math.Max(rh, p); rl = Math.Min(rl, p);
                for (int t = 0; t < 3; t++)
                {
                    if (!trig[t] && (rh - rl) >= xs[t] * atr)
                    {
                        trig[t] = true;
                        if (p >= rh - 0.25) o.Add(x.Emit(i, false, xs[t], "ATR09"));
                        else if (p <= rl + 0.25) o.Add(x.Emit(i, true, xs[t], "ATR09"));
                    }
                }
            }
            return o;
        }

        public static List<Fire> Pivot16(Ctx x)
        {
            if (x.Prior.Count == 0) return new List<Fire>();
            var d = x.Prior[x.Prior.Count - 1];
            double pp = (d.High + d.Low + d.Close) / 3.0;
            double s1 = 2 * pp - d.High, r1 = 2 * pp - d.Low;
            var idx = new List<int>(); foreach (int i in x.RthIdx()) idx.Add(i);
            if (idx.Count == 0) return new List<Fire>();
            double o0 = x.C[idx[0]]; var o = new List<Fire>(); bool g1 = false, g2 = false;
            foreach (int i in idx)
            {
                double p = x.C[i];
                if (!g1 && o0 > s1 && p <= s1) { o.Add(x.Emit(i, true, o0 - p, "PIVOT16")); g1 = true; }
                if (!g2 && o0 < r1 && p >= r1) { o.Add(x.Emit(i, false, p - o0, "PIVOT16")); g2 = true; }
            }
            return o;
        }

        public static List<Fire> Renko24(Ctx x)
        {
            const double B = 2.0; var o = new List<Fire>();
            bool havePrev = false; double prevClose = 0; int curD = 0, prevD = 0, chain = 0;
            foreach (int i in x.RthIdx())
            {
                double p = x.C[i];
                if (!havePrev) { prevClose = Math.Floor(p / B) * B; havePrev = true; continue; }
                while (true)
                {
                    if (curD == 0)
                    {
                        if (p >= prevClose + B) { curD = 1; prevD = 0; prevClose += B; chain = 1; }
                        else if (p <= prevClose - B) { curD = -1; prevD = 0; prevClose -= B; chain = 1; }
                        else break;
                    }
                    else if (curD == 1)
                    {
                        if (p >= prevClose + B) { prevClose += B; chain++; if (chain == 2 && prevD == -1) o.Add(x.Emit(i, true, B, "RENKO24")); }
                        else if (p <= prevClose - 2 * B) { prevD = 1; curD = -1; prevClose -= 2 * B; chain = 1; }
                        else break;
                    }
                    else
                    {
                        if (p <= prevClose - B) { prevClose -= B; chain++; if (chain == 2 && prevD == 1) o.Add(x.Emit(i, false, B, "RENKO24")); }
                        else if (p >= prevClose + 2 * B) { prevD = -1; curD = 1; prevClose += 2 * B; chain = 1; }
                        else break;
                    }
                }
            }
            return o;
        }

        public static List<Fire> Sar23(Ctx x)
        {
            int n = x.N; var bull = new bool[n]; var psar = new double[n];
            var h = x.H; var l = x.L;
            for (int i = 0; i < n; i++) bull[i] = true;
            psar[0] = l[0]; double ep = h[0], af = 0.02;
            for (int i = 1; i < n; i++)
            {
                if (bull[i - 1])
                {
                    double cur = psar[i - 1] + af * (ep - psar[i - 1]);
                    cur = i >= 2 ? Math.Min(cur, Math.Min(l[i - 1], l[i - 2])) : Math.Min(cur, l[i - 1]);
                    if (l[i] < cur) { bull[i] = false; psar[i] = ep; ep = l[i]; af = 0.02; }
                    else { bull[i] = true; psar[i] = cur; if (h[i] > ep) { ep = h[i]; af = Math.Min(af + 0.02, 0.2); } }
                }
                else
                {
                    double cur = psar[i - 1] - af * (psar[i - 1] - ep);
                    cur = i >= 2 ? Math.Max(cur, Math.Max(h[i - 1], h[i - 2])) : Math.Max(cur, h[i - 1]);
                    if (h[i] > cur) { bull[i] = true; psar[i] = ep; ep = h[i]; af = 0.02; }
                    else { bull[i] = false; psar[i] = cur; if (l[i] < ep) { ep = l[i]; af = Math.Min(af + 0.02, 0.2); } }
                }
            }
            var o = new List<Fire>(); int cool = 0;
            foreach (int i in x.RthIdx())
            {
                if (i < 1) continue;
                if (cool > 0) { cool--; continue; }
                if (bull[i] != bull[i - 1]) { o.Add(x.Emit(i, bull[i], Math.Abs(x.C[i] - psar[i]), "SAR23")); cool = COOLDOWN; }
            }
            return o;
        }

        public static List<Fire> Rsi06(Ctx x)
        {
            var delta = Pd.Diff(x.C);
            var up = new double[x.N]; var dn = new double[x.N];
            for (int i = 0; i < x.N; i++)
            {
                if (!Pd.Fin(delta[i])) { up[i] = double.NaN; dn[i] = double.NaN; continue; }
                up[i] = Math.Max(delta[i], 0.0);
                dn[i] = -Math.Min(delta[i], 0.0);
            }
            var ag = Pd.EwmCom(up, 167); var al = Pd.EwmCom(dn, 167);
            var rsi = new double[x.N];
            for (int i = 0; i < x.N; i++) rsi[i] = 100.0 - 100.0 / (1.0 + ag[i] / al[i]);
            var pmax = Pd.RollMax(x.C, 360, 360); var pmin = Pd.RollMin(x.C, 360, 360);
            var rmax = Pd.RollMax(rsi, 360, 360); var rmin = Pd.RollMin(rsi, 360, 360);
            var o = new List<Fire>(); int cool = 0;
            foreach (int i in x.RthIdx())
            {
                if (!Pd.Fin(pmax[i]) || !Pd.Fin(rmax[i])) continue;
                if (cool > 0) { cool--; continue; }
                if (x.C[i] == pmax[i] && rsi[i] < rmax[i]) { o.Add(x.Emit(i, false, rmax[i] - rsi[i], "RSI06")); cool = COOLDOWN; }
                else if (x.C[i] == pmin[i] && rsi[i] > rmin[i]) { o.Add(x.Emit(i, true, rsi[i] - rmin[i], "RSI06")); cool = COOLDOWN; }
            }
            return o;
        }

        public static List<Fire> Macd07(Ctx x)
        {
            var e1 = Pd.EwmSpan(x.C, 144); var e2 = Pd.EwmSpan(x.C, 312);
            var m = new double[x.N]; for (int i = 0; i < x.N; i++) m[i] = e1[i] - e2[i];
            var ph = Pd.RollMax(x.C, 360, 360); var pl = Pd.RollMin(x.C, 360, 360);
            var mh = Pd.RollMax(m, 360, 360); var ml = Pd.RollMin(m, 360, 360);
            var o = new List<Fire>(); int cool = 0;
            foreach (int i in x.RthIdx())
            {
                if (!Pd.Fin(ph[i]) || !Pd.Fin(mh[i])) continue;
                if (cool > 0) { cool--; continue; }
                if (x.C[i] >= ph[i] && m[i] < mh[i]) { o.Add(x.Emit(i, false, mh[i] - m[i], "MACD07")); cool = COOLDOWN; }
                else if (x.C[i] <= pl[i] && m[i] > ml[i]) { o.Add(x.Emit(i, true, m[i] - ml[i], "MACD07")); cool = COOLDOWN; }
            }
            return o;
        }

        public static List<Fire> CtxEr(Ctx x)
        {
            const int ER_N = 10; const double ER_CHOP = 0.30;
            var b = Buckets.Build(x.Ts, x.O, x.H, x.L, x.C, x.V, 60);
            int m = b.Ids.Length; var c = b.C;
            if (m < ER_N + 2) return new List<Fire>();
            var dc = new double[m]; dc[0] = 0;
            for (int k = 1; k < m; k++) dc[k] = Math.Abs(c[k] - c[k - 1]);
            var denom = Pd.RollSum(dc, ER_N, ER_N);
            var net = new double[m]; var er = new double[m];
            for (int k = 0; k < m; k++)
            {
                net[k] = k >= ER_N ? c[k] - c[k - ER_N] : double.NaN;
                double dd = denom[k];
                er[k] = (Pd.Fin(dd) && dd > 0) ? Math.Abs(net[k]) / dd : double.NaN;
            }
            var o = new List<Fire>();
            for (int k = ER_N + 1; k < m; k++)
            {
                if (!Pd.Fin(er[k]) || !Pd.Fin(er[k - 1])) continue;
                if (!(er[k] < ER_CHOP && er[k - 1] >= ER_CHOP)) continue;
                if (!Pd.Fin(net[k]) || net[k] == 0) continue;
                int r = BktRow(x, b, k); if (r < 0) continue;
                o.Add(x.Emit(r, net[k] < 0, er[k], "CTXER"));
            }
            return o;
        }

        public static List<Fire> ExitKmdr(Ctx x)
        {
            const int L_LO = 28, L_HI = 22, MOM_LO = 6, MOM_HI = 11, ATR_N = 14; const double ATRS = 1.5;
            var b = Buckets.Build(x.Ts, x.O, x.H, x.L, x.C, x.V, 60);
            int m = b.Ids.Length; var c = b.C; var h = b.H; var l = b.L;
            var pc = Pd.Shift(c, 1); var tr = new double[m];
            for (int k = 0; k < m; k++)
            {
                double a = h[k] - l[k], mx = a;
                if (Pd.Fin(pc[k])) { mx = Math.Max(mx, Math.Abs(h[k] - pc[k])); mx = Math.Max(mx, Math.Abs(l[k] - pc[k])); }
                tr[k] = mx;
            }
            var atr = Pd.EwmAlpha(tr, 1.0 / ATR_N);
            var emaLo = Pd.EwmSpan(c, L_LO); var emaHi = Pd.EwmSpan(c, L_HI);
            var momLo = new double[m]; var momHi = new double[m];
            for (int k = 0; k < m; k++)
            {
                momLo[k] = k >= MOM_LO ? c[k] - c[k - MOM_LO] : double.NaN;
                momHi[k] = k >= MOM_HI ? c[k] - c[k - MOM_HI] : double.NaN;
            }
            var accLo = new double[m]; var accHi = new double[m]; accLo[0] = double.NaN; accHi[0] = double.NaN;
            for (int k = 1; k < m; k++)
            {
                accLo[k] = (Pd.Fin(momLo[k]) && Pd.Fin(momLo[k - 1])) ? momLo[k] - momLo[k - 1] : double.NaN;
                accHi[k] = (Pd.Fin(momHi[k]) && Pd.Fin(momHi[k - 1])) ? momHi[k] - momHi[k - 1] : double.NaN;
            }
            var condL = new bool[m]; var condS = new bool[m];
            for (int k = 0; k < m; k++)
            {
                bool lower = c[k] <= emaLo[k] - ATRS * atr[k];
                bool upper = c[k] >= emaHi[k] + ATRS * atr[k];
                condL[k] = lower && momLo[k] < 0 && accLo[k] < 0;
                condS[k] = upper && momHi[k] > 0 && accHi[k] > 0;
            }
            var o = new List<Fire>();
            for (int k = 1; k < m; k++)
            {
                bool fl = condL[k] && !condL[k - 1]; bool fs = condS[k] && !condS[k - 1];
                if (!(fl || fs)) continue;
                int r = BktRow(x, b, k); if (r < 0) continue;
                if (fl) o.Add(x.Emit(r, true, Math.Abs(momLo[k]), "EXITKMDR"));
                if (fs) o.Add(x.Emit(r, false, Math.Abs(momHi[k]), "EXITKMDR"));
            }
            return o;
        }

        public static List<Fire> PtrnEngulf(Ctx x)
        {
            var b = Buckets.Build(x.Ts, x.O, x.H, x.L, x.C, x.V, 60);
            int m = b.Ids.Length; var o = b.O; var h = b.H; var l = b.L; var c = b.C;
            var po = Pd.Shift(o, 1); var pc = Pd.Shift(c, 1);
            var res = new List<Fire>();
            for (int k = 1; k < m; k++)
            {
                double body = Math.Abs(c[k] - o[k]);
                double rng = (h[k] - l[k]) == 0 ? 1e-10 : h[k] - l[k];
                bool doji = body / rng < 0.1;
                double upper = h[k] - Math.Max(c[k], o[k]);
                double lower = Math.Min(c[k], o[k]) - l[k];
                bool hammer = !doji && lower > 2.0 * body && upper < 0.1 * rng && body < 0.3 * rng;
                bool ebull = !doji && !hammer && pc[k] < po[k] && c[k] > o[k] && o[k] <= pc[k] && c[k] >= po[k];
                bool ebear = !doji && !hammer && pc[k] > po[k] && c[k] < o[k] && o[k] >= pc[k] && c[k] <= po[k];
                if (!(ebull || ebear)) continue;
                int r = b.CloseRow[k]; if (r < 0 || r < x.Start || !x.Rth[r]) continue;
                res.Add(x.Emit(r, ebull, body, "PTRNENGULF"));
            }
            return res;
        }

        const double Z_ENTRY = 1.8481, Z_EXIT = 0.4752, NMP_EPS = 0.1; const int NMP_K = 21;

        static double[] Vr1m(Ctx x)
        {
            var b = Buckets.Build(x.Ts, x.O, x.H, x.L, x.C, x.V, 60);
            int m = b.Ids.Length; var c = b.C;
            var s10 = Pd.RollStd(c, 10, 10, 1); var s60 = Pd.RollStd(c, 60, 60, 1);
            var vrb = new double[m];
            for (int k = 0; k < m; k++) { double a = s60[k]; vrb[k] = (Pd.Fin(a) && a != 0) ? s10[k] / a : double.NaN; }
            var outv = new double[x.N];
            for (int i = 0; i < x.N; i++)
            {
                long want = x.Ts[i] / 60 - 1;
                int pp;
                outv[i] = b.Pos.TryGetValue(want, out pp) ? vrb[pp] : double.NaN;
            }
            return outv;
        }

        static List<NmpFire> NmpFires(Ctx x)
        {
            var o = new List<NmpFire>(); bool armed = true;
            for (int i = 0; i < x.N; i++)
            {
                double zi = x.Zse[i]; if (!Pd.Fin(zi)) continue;
                if (Math.Abs(zi) < Z_EXIT) armed = true;
                else if (armed && Math.Abs(zi) > Z_ENTRY) { armed = false; if (x.Rth[i] && i >= x.Start) { var f = new NmpFire(); f.I = i; f.Z = zi; o.Add(f); } }
            }
            return o;
        }

        public static double[] NmpLambda(Ctx x)
        {
            var mrows = new List<int>();
            for (int i = 0; i < x.N; i++) if (x.Ts[i] % 60 == 0) mrows.Add(i);
            int mm = mrows.Count; var logz = new double[mm];
            for (int k = 0; k < mm; k++)
            {
                double z = x.Zse[mrows[k]];
                logz[k] = Pd.Fin(z) ? Math.Log(Math.Abs(z) + NMP_EPS) : double.NaN;
            }
            var lam1m = new double[mm]; for (int k = 0; k < mm; k++) lam1m[k] = double.NaN;
            if (mm >= NMP_K)
            {
                double xm = (NMP_K - 1) / 2.0, xv = 0;
                for (int t = 0; t < NMP_K; t++) { double dx = t - xm; xv += dx * dx; }
                var w = new double[NMP_K]; for (int t = 0; t < NMP_K; t++) w[t] = (t - xm) / xv;
                for (int k = NMP_K - 1; k < mm; k++)
                {
                    double dot = 0; bool ok = true;
                    for (int t = 0; t < NMP_K; t++) { double val = logz[k - NMP_K + 1 + t]; if (!Pd.Fin(val)) { ok = false; break; } dot += val * w[t]; }
                    lam1m[k] = ok ? dot : double.NaN;
                }
            }
            var lam = new double[x.N]; for (int i = 0; i < x.N; i++) lam[i] = double.NaN;
            for (int k = 0; k < mm; k++) lam[mrows[k]] = lam1m[k];
            double last = double.NaN;
            for (int i = 0; i < x.N; i++) { if (Pd.Fin(lam[i])) last = lam[i]; else lam[i] = last; }
            return lam;
        }

        public static List<Fire> Nmp(Ctx x)
        {
            if (!x.HasZse) return new List<Fire>();
            var vr = Vr1m(x); var o = new List<Fire>();
            foreach (var f in NmpFires(x))
                if (Pd.Fin(vr[f.I]) && vr[f.I] < 1.0) o.Add(x.Emit(f.I, f.Z < 0, Math.Abs(f.Z), "NMP"));
            return o;
        }

        static List<Nmp9Ev> Nmp9Events(Ctx x)
        {
            var ev = new List<Nmp9Ev>();
            var m1 = TfState.Build(x, 60); var m5 = TfState.Build(x, 300);
            var m15 = TfState.Build(x, 900); var h1 = TfState.Build(x, 3600);
            double[] lam = x.HasZse ? NmpLambda(x) : NullArr(x.N);
            const double ROCHE = 2.0, VR_ENTRY = 1.0, WICK5 = 0.83, WICK15 = 0.77,
                VELT = 50.0, FREIGHT = 100.0, H1Z = 1.0, H1AG = 1.5;
            string prev = null;
            for (int i = 0; i < x.N; i++)
            {
                if (!(x.Ts[i] % 60 == 0 && x.Rth[i] && i >= x.Start)) continue;
                int k1 = m1.At(i), k5 = m5.At(i), k15 = m15.At(i), kh = h1.At(i);
                if (k1 < 0 || k5 < 0 || k15 < 0 || kh < 0) continue;
                double z = m1.Z[k1], vr = m1.Vr[k1];
                if (!(Pd.Fin(z) && Math.Abs(z) > ROCHE && Pd.Fin(vr) && vr < VR_ENTRY)) { prev = null; continue; }
                bool longDir = z <= 0;
                double wick5 = m5.Wick[k5], wick15 = m15.Wick[k15];
                double h1z = h1.Z[kh], h1vel = h1.Vel[kh];
                double vel = m1.Vel[k1], absVel = Math.Abs(vel);
                bool hasWick = wick5 > WICK5 && wick15 > WICK15;
                bool h1Aligned = (longDir && h1z < -H1Z) || (!longDir && h1z > H1Z);
                bool h1AgainstFade = (longDir && h1z > H1AG) || (!longDir && h1z < -H1AG);
                bool h1VelAgainst = (longDir && h1vel < -H1AG) || (!longDir && h1vel > H1AG);
                string tier = null; bool rl = false; double val = 0;
                if (hasWick && h1Aligned) { tier = "CASCADE"; rl = longDir; val = wick5; }
                else if (hasWick) { tier = "KILLSHOT"; rl = longDir; val = wick5; }
                else if (absVel >= FREIGHT) { tier = "FREIGHT"; rl = vel > 0; val = absVel; }
                else if (h1AgainstFade) { tier = "FADEAGAINST"; rl = !(h1z > 0); val = Math.Abs(h1z); }
                else if (h1VelAgainst) { tier = "RIDEAGAINST"; rl = h1vel > 0; val = Math.Abs(h1vel); }
                else
                {
                    double lamI = lam[i];
                    if (Pd.Fin(lamI) && lamI > 0.0)
                    {
                        bool rdirLong = !longDir;
                        if (absVel >= VELT) { tier = "RIDEMOM"; rl = rdirLong; val = absVel; }
                        else { tier = "RIDECALM"; rl = rdirLong; val = Math.Abs(z); }
                    }
                    else if (absVel >= VELT) { tier = "FADEMOM"; rl = longDir; val = absVel; }
                    else { tier = "FADECALM"; rl = longDir; val = Math.Abs(z); }
                }
                string key = tier == null ? null : (rl ? "long" : "short") + "|" + tier;
                if (tier != null && key != prev) { var e = new Nmp9Ev(); e.I = i; e.IsLong = rl; e.Tier = "NMP9" + tier; e.Val = val; ev.Add(e); }
                prev = key;
            }
            return ev;
        }

        public static List<Fire> Nmp9(Ctx x, string tier)
        {
            var o = new List<Fire>();
            foreach (var e in Nmp9Events(x))
                if (e.Tier == "NMP9" + tier) o.Add(x.Emit(e.I, e.IsLong, e.Val, "NMP9" + tier));
            return o;
        }

        static List<Nmp9Ev> NmptEvents(Ctx x)
        {
            var ev = new List<Nmp9Ev>();
            var m1 = TfState.Build(x, 60); var m5 = TfState.Build(x, 300);
            var m15 = TfState.Build(x, 900); var h1 = TfState.Build(x, 3600);
            const double WICK5 = 0.83, WICK15 = 0.77, H1Z = 1.0, H1AG = 1.5, FREIGHT = 100.0,
                FVRMAX = 0.85, MTF5VEL = 30.0, MTF1VEL = 10.0, MTFZ = 1.4, MTFVR = 0.58, MTFVOL = 2.0;
            string prev = null;
            for (int i = 0; i < x.N; i++)
            {
                if (!(x.Ts[i] % 60 == 0 && x.Rth[i] && i >= x.Start)) continue;
                int k1 = m1.At(i), k5 = m5.At(i), k15 = m15.At(i), kh = h1.At(i);
                if (k1 < 0 || k5 < 0 || k15 < 0 || kh < 0) continue;
                double z = m1.Z[k1];
                if (!Pd.Fin(z)) continue;
                bool longDir = z <= 0;
                double wick5 = m5.Wick[k5], wick15 = m15.Wick[k15];
                double h1z = h1.Z[kh], h1vel = h1.Vel[kh];
                double vel = m1.Vel[k1], acc = m1.Acc[k1], absVel = Math.Abs(vel);
                double vr = m1.Vr[k1], v5vel = m5.Vel[k5], v5acc = m5.Acc[k5];
                double dmi = m1.Dmi[k1], volRel = m1.Volr[k1];
                double z5 = Math.Abs(m5.Z[k5]), z15 = Math.Abs(m15.Z[k15]);
                bool hasWick = wick5 > WICK5 && wick15 > WICK15;
                bool h1AgainstFade = (longDir && h1z > H1AG) || (!longDir && h1z < -H1AG);
                bool h1Aligned = (longDir && h1z < -H1Z) || (!longDir && h1z > H1Z);
                string tier = null; bool rl = false; double val = 0;
                if (Pd.Fin(vr) && absVel >= FREIGHT && vel * acc > 0 && vr < FVRMAX)
                { tier = "FREIGHT"; rl = vel > 0; val = absVel; }
                else if (hasWick && !h1Aligned) { tier = "KILLSHOT"; rl = longDir; val = wick5; }
                else if (hasWick && h1Aligned) { tier = "CASCADE"; rl = longDir; val = wick5; }
                else if (((longDir && h1vel < -3.0) || (!longDir && h1vel > 3.0)) && !h1AgainstFade)
                { tier = "RIDEAGN"; rl = h1vel > 0; val = Math.Abs(h1vel); }
                else if (h1AgainstFade && Math.Abs(v5vel) < 10.0) { tier = "FADEAGN"; rl = longDir; val = Math.Abs(h1z); }
                else if (Pd.Fin(vr) && Pd.Fin(volRel) && v5acc < 0 && Math.Abs(v5vel) > MTF5VEL &&
                         absVel > MTF1VEL && Math.Abs(z) > MTFZ && vr > MTFVR && volRel > MTFVOL)
                { tier = "MTFEXH"; rl = v5vel > 0; val = Math.Abs(v5vel); }
                else if (z5 > 1.3 && z15 > 1.3)
                {
                    bool bdirLong = z > 0;
                    if ((bdirLong && dmi > -5) || (!bdirLong && dmi < 5)) { tier = "MTFBRK"; rl = bdirLong; val = Math.Min(z5, z15); }
                }
                else
                {
                    bool hiOpp = (longDir && v5vel < -3 && h1vel < -3) || (!longDir && v5vel > 3 && h1vel > 3);
                    if (!hiOpp) { tier = "FADECALM"; rl = longDir; val = Math.Abs(z); }
                }
                string key = tier == null ? null : (rl ? "long" : "short") + "|" + tier;
                if (tier != null && key != prev) { var e = new Nmp9Ev(); e.I = i; e.IsLong = rl; e.Tier = "NMPT" + tier; e.Val = val; ev.Add(e); }
                prev = key;
            }
            return ev;
        }

        public static List<Fire> Nmpt(Ctx x, string tier)
        {
            var o = new List<Fire>();
            foreach (var e in NmptEvents(x))
                if (e.Tier == "NMPT" + tier) o.Add(x.Emit(e.I, e.IsLong, e.Val, "NMPT" + tier));
            return o;
        }

        static int BktRow(Ctx x, Buckets b, int k)
        {
            int r = b.CloseRow[k];
            if (r < 0 || r < x.Start || !x.Rth[r]) return -1;
            return r;
        }
        static double[] NullArr(int n) { var a = new double[n]; for (int i = 0; i < n; i++) a[i] = double.NaN; return a; }
    }

    // ---- TMPL0 frozen-codebook stream (Tmpl0.cs, verbatim; embedded codebook) ----
    class Tmpl0
    {
        double[] mean, scale;
        double[][] Cs;
        double[] longFrac;
        int[] memberCount;
        const double TICK = 0.25;
        const int HURST_N = 30;
        const int MIN_MEMBERS = 20;
        const double MIN_CONV = 0.05;
        static readonly long[] PERIODS = { 60, 300, 900 };

        public Tmpl0()
        {
            mean = Tmpl0Data.ScalerMean; scale = Tmpl0Data.ScalerScale;
            int nt = Tmpl0Data.NTemplates;
            Cs = new double[nt][]; longFrac = new double[nt]; memberCount = new int[nt];
            for (int t = 0; t < nt; t++)
            {
                var cs = new double[6];
                for (int d = 0; d < 6; d++)
                {
                    double cen = Tmpl0Data.Centroids[t * 6 + d];
                    cs[d] = (cen - mean[d]) / scale[d];
                }
                Cs[t] = cs;
                memberCount[t] = Tmpl0Data.MemberCount[t];
                longFrac[t] = Tmpl0Data.LongFrac[t];
            }
        }

        struct Ev { public int Row; public long Ts; public double PivAge, Tod; public int Leg; public double[] F; public long Tf; }

        public List<Fire> Run(Ctx x)
        {
            var events = new List<Ev>();
            foreach (long period in PERIODS)
            {
                var b = Buckets.Build(x.Ts, x.O, x.H, x.L, x.C, x.V, period);
                int m = b.Ids.Length;
                if (m < HURST_N + 2) continue;
                var zAbs = MathX.Z21(b.C);
                var velFeat = new double[m]; velFeat[0] = double.NaN;
                for (int k = 1; k < m; k++) velFeat[k] = Math.Log(1.0 + Math.Abs((b.C[k] - b.C[k - 1]) / TICK));
                double tfFeat = Pd.Log2(Math.Max(1, period));
                double[] dmi, adx;
                WilderDmiAdx(b.H, b.L, b.C, out dmi, out adx);
                var hurst = RsHurst(b.C, HURST_N);
                var cdl = CandleFlags(b.O, b.H, b.L, b.C);
                var geo = GeoFlags(b.H, b.L);
                for (int k = 0; k < m; k++)
                {
                    if (cdl[k] == 0 && geo[k] == 0) continue;
                    var fv = new double[6] {
                        Math.Abs(zAbs[k]), velFeat[k], tfFeat, adx[k] / 100.0, hurst[k], dmi[k] / 100.0
                    };
                    bool ok = true; foreach (var f in fv) if (!Pd.Fin(f)) { ok = false; break; }
                    if (!ok) continue;
                    int r = b.CloseRow[k];
                    if (r < 0 || r < x.Start || !x.Rth[r]) continue;
                    var ev = new Ev();
                    ev.Row = r; ev.Ts = x.Ts[r]; ev.PivAge = (r - x.PivI[r]) * 5.0 / 60.0;
                    ev.Tod = x.Tod[r]; ev.Leg = x.Leg[r]; ev.F = fv; ev.Tf = period;
                    if (cdl[k] != 0) events.Add(ev);
                    if (geo[k] != 0) events.Add(ev);
                }
            }
            var o = new List<Fire>();
            foreach (var ev in events)
            {
                var xs = new double[6];
                for (int d = 0; d < 6; d++) xs[d] = (ev.F[d] - mean[d]) / scale[d];
                int tid = Nearest(xs);
                double lf = longFrac[tid]; int mc = memberCount[tid];
                double conv = Math.Abs(lf - 0.5);
                if (!(mc >= MIN_MEMBERS && Pd.Fin(lf) && conv >= MIN_CONV)) continue;
                bool isLong = lf > 0.5;
                int swl = ev.Leg != 0 ? ((ev.Leg > 0) == isLong ? 1 : 0) : 0;
                var f = new Fire();
                f.Row = ev.Row; f.Ts = ev.Ts; f.IsLong = isLong; f.Value = conv;
                f.PivotAgeMin = ev.PivAge; f.SigWithLeg = swl; f.Tod = ev.Tod; f.Det = "TMPL0";
                f.Tf = ev.Tf;
                o.Add(f);
            }
            return o;
        }

        int Nearest(double[] xs)
        {
            int best = 0; double bd = double.PositiveInfinity;
            for (int t = 0; t < Cs.Length; t++)
            {
                var c = Cs[t]; double d = 0;
                for (int j = 0; j < 6; j++) { double e = xs[j] - c[j]; d += e * e; }
                if (d < bd) { bd = d; best = t; }
            }
            return best;
        }

        static void WilderDmiAdx(double[] h, double[] l, double[] c, out double[] dmiOut, out double[] adxOut)
        {
            int n = h.Length;
            var up = Pd.Diff(h); var dnRaw = Pd.Diff(l);
            var dmp = new double[n]; var dmm = new double[n];
            for (int i = 0; i < n; i++)
            {
                double u = up[i], dn = -dnRaw[i];
                dmp[i] = (Pd.Fin(u) && Pd.Fin(dn) && u > dn && u > 0) ? u : 0.0;
                dmm[i] = (Pd.Fin(u) && Pd.Fin(dn) && dn > u && dn > 0) ? dn : 0.0;
            }
            var pc = Pd.Shift(c, 1); var tr = new double[n];
            for (int i = 0; i < n; i++)
            {
                double a = h[i] - l[i], mx = a;
                if (Pd.Fin(pc[i])) { mx = Math.Max(mx, Math.Abs(h[i] - pc[i])); mx = Math.Max(mx, Math.Abs(l[i] - pc[i])); }
                tr[i] = mx;
            }
            var trs = Pd.EwmAlpha(tr, 1.0 / 14.0);
            var dips = Pd.EwmAlpha(dmp, 1.0 / 14.0);
            var dims = Pd.EwmAlpha(dmm, 1.0 / 14.0);
            var dip = new double[n]; var dim = new double[n]; var dmi = new double[n]; var dx = new double[n];
            for (int i = 0; i < n; i++)
            {
                double t = trs[i];
                dip[i] = (Pd.Fin(t) && t != 0) ? 100.0 * dips[i] / t : double.NaN;
                dim[i] = (Pd.Fin(t) && t != 0) ? 100.0 * dims[i] / t : double.NaN;
                dmi[i] = dip[i] - dim[i];
                double sum = dip[i] + dim[i];
                dx[i] = (Pd.Fin(sum) && sum != 0) ? 100.0 * Math.Abs(dip[i] - dim[i]) / sum : double.NaN;
            }
            adxOut = Pd.EwmAlpha(dx, 1.0 / 14.0);
            dmiOut = dmi;
        }

        static double[] RsHurst(double[] c, int N)
        {
            int n = c.Length; var H = new double[n]; for (int i = 0; i < n; i++) H[i] = double.NaN;
            if (n < N) return H;
            for (int k = N - 1; k < n; k++)
            {
                int s = k - N + 1; double mean = 0;
                for (int j = 0; j < N; j++) mean += c[s + j]; mean /= N;
                double cum = 0, mx = double.NegativeInfinity, mn = double.PositiveInfinity, ss = 0;
                for (int j = 0; j < N; j++)
                {
                    double yc = c[s + j] - mean; cum += yc;
                    if (cum > mx) mx = cum; if (cum < mn) mn = cum;
                    ss += yc * yc;
                }
                double R = mx - mn; double S = Math.Sqrt(ss / N);
                double rs = S > 0 ? R / S : double.NaN;
                double h = (Pd.Fin(rs) && rs > 0) ? Math.Log(rs) / Math.Log(N) : double.NaN;
                if (Pd.Fin(h)) H[k] = Math.Min(1.0, Math.Max(0.0, h));
                else H[k] = double.NaN;
            }
            return H;
        }

        static int[] CandleFlags(double[] o, double[] h, double[] l, double[] c)
        {
            int n = c.Length; var code = new int[n];
            var po = Pd.Shift(o, 1); var pc = Pd.Shift(c, 1);
            for (int k = 0; k < n; k++)
            {
                double body = Math.Abs(c[k] - o[k]);
                double rng = (h[k] - l[k]) == 0 ? 1e-10 : h[k] - l[k];
                double upper = h[k] - Math.Max(c[k], o[k]);
                double lower = Math.Min(c[k], o[k]) - l[k];
                bool doji = body / rng < 0.1;
                bool hammer = !doji && lower > 2.0 * body && upper < 0.1 * rng && body < 0.3 * rng;
                bool ebull = !doji && !hammer && pc[k] < po[k] && c[k] > o[k] && o[k] <= pc[k] && c[k] >= po[k];
                bool ebear = !doji && !hammer && pc[k] > po[k] && c[k] < o[k] && o[k] >= pc[k] && c[k] <= po[k];
                if (doji) code[k] = 1; else if (hammer) code[k] = 2; else if (ebull) code[k] = 3; else if (ebear) code[k] = 4;
            }
            return code;
        }

        static int[] GeoFlags(double[] h, double[] l)
        {
            int n = h.Length;
            var recMax = Pd.RollMax(h, 5, 5); var recMin = Pd.RollMin(l, 5, 5);
            var recRange = new double[n]; for (int i = 0; i < n; i++) recRange[i] = recMax[i] - recMin[i];
            var prevRange = Pd.Shift(recRange, 5);
            var l4 = Pd.Shift(l, 4); var h4 = Pd.Shift(h, 4);
            var l1 = Pd.Shift(l, 1); var prev4min = Pd.RollMin(l1, 4, 4);
            var code = new int[n];
            for (int i = 0; i < n; i++)
            {
                bool comp = Pd.Fin(prevRange[i]) && prevRange[i] > 0 && Pd.Fin(recRange[i]) && recRange[i] < prevRange[i] * 0.7;
                bool wedge = Pd.Fin(l4[i]) && Pd.Fin(h4[i]) && l[i] > l4[i] && h[i] < h4[i];
                bool brk = Pd.Fin(prev4min[i]) && l[i] < prev4min[i];
                if (comp) code[i] = 1; else if (wedge) code[i] = 2; else if (brk) code[i] = 3;
            }
            for (int i = 0; i < Math.Min(9, n); i++) code[i] = 0;
            return code;
        }
    }

    // ---- per-1m-bar decision record + core aggregation (Program.cs, verbatim) ----
    class BarRec
    {
        public long BarTs; public Dictionary<string, int> F = new Dictionary<string, int>();
        public string Gov = ""; public int GovDir = 0; public double P = double.NaN;
        public int Entry = 0; public int EntryDir = 0;
        public int ZzLeg = 0; public int ZzConfirm = 0;
        public double ZzPivAge = 0.0; public double ZzPivPrice = 0.0;
        public int LastRow = -1;
    }

    static class Core
    {
        // TMPL0 same-bar tie rule (P2 pin, doc 133): highest-TF wins; tie -> larger
        // conviction |long_frac-0.5|; still tied -> 0 (hold prior).
        public static int ResolveTmpl0(List<TmplSub> fs)
        {
            if (fs.Count == 0) return 0;
            long bestTf = long.MinValue; foreach (var e in fs) if (e.Tf > bestTf) bestTf = e.Tf;
            double bestConv = double.NegativeInfinity;
            foreach (var e in fs) if (e.Tf == bestTf && e.Conv > bestConv) bestConv = e.Conv;
            int dir = 0; bool set = false, tie = false;
            foreach (var e in fs)
                if (e.Tf == bestTf && e.Conv == bestConv)
                {
                    if (!set) { dir = e.Dir; set = true; }
                    else if (e.Dir != dir) tie = true;
                }
            return tie ? 0 : dir;
        }

        // (D4) stable index sort by Ts, tie-break by original index (== LINQ OrderBy.ThenBy).
        static int[] StableOrderByTs(List<Fire> fires)
        {
            int nf = fires.Count;
            var order = new int[nf];
            for (int i = 0; i < nf; i++) order[i] = i;
            Array.Sort(order, delegate (int a, int b)
            {
                long ta = fires[a].Ts, tb = fires[b].Ts;
                if (ta < tb) return -1; if (ta > tb) return 1;
                return a < b ? -1 : (a > b ? 1 : 0);
            });
            return order;
        }

        public static List<BarRec> ProcessDay(Ctx x, Tmpl0 tmpl)
        {
            var fires = new List<Fire>();
            fires.AddRange(Gens.Zigzag(x)); fires.AddRange(Gens.Orb02(x)); fires.AddRange(Gens.Vwap03(x)); fires.AddRange(Gens.Pivot16(x));
            fires.AddRange(Gens.Round05(x)); fires.AddRange(Gens.Dow19(x)); fires.AddRange(Gens.Tunnel20(x)); fires.AddRange(Gens.Atr09(x));
            fires.AddRange(Gens.Sar23(x)); fires.AddRange(Gens.Rsi06(x)); fires.AddRange(Gens.Macd07(x)); fires.AddRange(Gens.Renko24(x));
            fires.AddRange(Gens.Nmpt(x, "FADECALM")); fires.AddRange(Gens.Nmpt(x, "MTFBRK"));
            fires.AddRange(Gens.Nmp(x)); fires.AddRange(Gens.PtrnEngulf(x));
            fires.AddRange(Gens.CtxEr(x)); fires.AddRange(Gens.ExitKmdr(x));
            fires.AddRange(Gens.Nmp9(x, "RIDEAGAINST")); fires.AddRange(Gens.Nmp9(x, "RIDECALM")); fires.AddRange(Gens.Nmp9(x, "FADEAGAINST"));
            fires.AddRange(tmpl.Run(x));

            int nf = fires.Count;
            var order = StableOrderByTs(fires);
            var sorted = new List<Fire>(nf);
            for (int k = 0; k < nf; k++) sorted.Add(fires[order[k]]);
            var ts = new long[nf];
            for (int k = 0; k < nf; k++) ts[k] = sorted[k].Ts;
            long CONSENSUS_S = ModelData.ConsensusS;
            var cons = new int[nf];
            for (int k = 0; k < nf; k++)
            {
                long lo = ts[k] - CONSENSUS_S, hi = ts[k] + CONSENSUS_S;
                int a = LowerBound(ts, lo), bb = UpperBound(ts, hi);
                int sameDir = 0, own = 0; bool lng = sorted[k].IsLong; string det = sorted[k].Det;
                for (int j = a; j < bb; j++)
                    if (sorted[j].IsLong == lng) { sameDir++; if (sorted[j].Det == det) own++; }
                cons[k] = sameDir - own;
            }
            var P = new double[nf];
            for (int k = 0; k < nf; k++) P[k] = CompactP(sorted[k], cons[k]);

            var zzr = ZigzagRTrigger(x);

            var barMap = new SortedDictionary<long, BarRec>();
            for (int i = 0; i < x.N; i++)
                if (x.Rth[i] && i >= x.Start)
                {
                    long T = (x.Ts[i] / 60) * 60;
                    BarRec bar;
                    if (!barMap.TryGetValue(T, out bar))
                    {
                        bar = new BarRec(); bar.BarTs = T;
                        foreach (var d in ModelData.Topk) bar.F[d] = 0;
                        barMap[T] = bar;
                    }
                    bar.LastRow = i;
                    if (zzr.Flip[i] != 0) bar.ZzConfirm = zzr.Flip[i];
                }
            var tmplByBar = new Dictionary<long, List<TmplSub>>();
            for (int k = 0; k < nf; k++)
            {
                long T = (sorted[k].Ts / 60) * 60;
                BarRec bar;
                if (!barMap.TryGetValue(T, out bar)) continue;
                string det = sorted[k].Det;
                if (det == "TMPL0")
                {
                    List<TmplSub> lst;
                    if (!tmplByBar.TryGetValue(T, out lst)) { lst = new List<TmplSub>(); tmplByBar[T] = lst; }
                    var sub = new TmplSub(); sub.Tf = sorted[k].Tf; sub.Conv = sorted[k].Value; sub.Dir = sorted[k].IsLong ? 1 : -1;
                    lst.Add(sub);
                }
                else
                {
                    bar.F[det] = sorted[k].IsLong ? 1 : -1;
                }
                double p = P[k];
                if (Pd.Fin(p) && (!Pd.Fin(bar.P) || p > bar.P))
                { bar.P = p; bar.Gov = det; bar.GovDir = sorted[k].IsLong ? 1 : -1; }
            }
            foreach (var bar in barMap.Values)
            {
                List<TmplSub> lst;
                if (bar.F.ContainsKey("TMPL0") && tmplByBar.TryGetValue(bar.BarTs, out lst))
                    bar.F["TMPL0"] = ResolveTmpl0(lst);
                bar.Entry = (Pd.Fin(bar.P) && bar.P >= ModelData.Threshold) ? 1 : 0;
                bar.EntryDir = bar.Entry == 1 ? bar.GovDir : 0;
                int r = bar.LastRow;
                if (r >= 0)
                {
                    bar.ZzLeg = zzr.Dir[r];
                    bar.ZzPivAge = (r - zzr.PivBar[r]) * 5.0 / 60.0;
                    bar.ZzPivPrice = zzr.PivPx[r];
                }
            }
            var outl = new List<BarRec>();
            foreach (var bar in barMap.Values) outl.Add(bar);
            return outl;
        }

        public static ZzResult ZigzagRTrigger(Ctx x)
        {
            double TICK = Ctx.TICK;
            const int MIN_BARS_5S = 36;
            int n = x.N;
            var priceT = new double[n];
            for (int i = 0; i < n; i++) priceT[i] = x.C[i] / TICK;
            int firstRth = x.Start;
            for (int i = 0; i < n; i++) if (x.Rth[i] && i >= x.Start) { firstRth = i; break; }
            double thrPts = x.ZzThr[firstRth];
            if (!Pd.Fin(thrPts))
                for (int i = firstRth; i < n; i++) if (Pd.Fin(x.ZzThr[i])) { thrPts = x.ZzThr[i]; break; }
            int minRev = Math.Max(4, (int)Math.Round(thrPts / TICK, MidpointRounding.ToEven));

            var dir = new int[n]; var flip = new int[n];
            var pivBar = new int[n]; var pivPx = new double[n];
            int d = 0;
            double ext = priceT[0]; int extBar = 0;
            double firstClose = priceT[0];
            int lastPivBar = 0; double lastPivPx = priceT[0];
            for (int i = 1; i < n; i++)
            {
                double p = priceT[i]; int f = 0;
                if (d == 0)
                {
                    if (p > ext) { ext = p; extBar = i; }
                    if (p < firstClose && (firstClose - p) >= minRev) { d = -1; ext = p; extBar = i; f = -1; }
                    else if (p > firstClose && (p - firstClose) >= minRev) { d = 1; ext = p; extBar = i; f = 1; }
                    if (f != 0) { lastPivBar = i; lastPivPx = firstClose; }
                }
                else if (d == 1)
                {
                    if (p >= ext) { ext = p; extBar = i; }
                    else if ((ext - p) >= minRev && (i - extBar) >= MIN_BARS_5S)
                    { lastPivBar = extBar; lastPivPx = ext; d = -1; ext = p; extBar = i; f = -1; }
                }
                else
                {
                    if (p <= ext) { ext = p; extBar = i; }
                    else if ((p - ext) >= minRev && (i - extBar) >= MIN_BARS_5S)
                    { lastPivBar = extBar; lastPivPx = ext; d = 1; ext = p; extBar = i; f = 1; }
                }
                dir[i] = d; flip[i] = f; pivBar[i] = lastPivBar; pivPx[i] = lastPivPx * TICK;
            }
            var res = new ZzResult();
            res.Dir = dir; res.Flip = flip; res.PivBar = pivBar; res.PivPx = pivPx; res.MinRev = minRev;
            return res;
        }

        public static double CompactP(Fire f, int consensus)
        {
            var cols = ModelData.Cols; var mu = ModelData.Mu; var sd = ModelData.Sd; var coef = ModelData.Coef;
            int nc = cols.Length; double logit = 0;
            for (int ci = 0; ci < nc; ci++)
            {
                string col = cols[ci]; double xv;
                switch (col)
                {
                    case "pivot_age_min": xv = f.PivotAgeMin; break;
                    case "sig_with_leg": xv = f.SigWithLeg; break;
                    case "tod": xv = f.Tod; break;
                    case "inter": xv = f.SigWithLeg * f.PivotAgeMin; break;
                    case "consensus": xv = consensus; break;
                    default: xv = (col == "is_" + f.Det) ? 1.0 : 0.0; break;
                }
                double z = (xv - mu[ci]) / sd[ci];
                logit += z * coef[ci];
            }
            return 1.0 / (1.0 + Math.Exp(-logit));
        }

        public static int LowerBound(long[] a, long v)
        {
            int lo = 0, hi = a.Length;
            while (lo < hi) { int mid = (lo + hi) >> 1; if (a[mid] < v) lo = mid + 1; else hi = mid; }
            return lo;
        }
        public static int UpperBound(long[] a, long v)
        {
            int lo = 0, hi = a.Length;
            while (lo < hi) { int mid = (lo + hi) >> 1; if (a[mid] <= v) lo = mid + 1; else hi = mid; }
            return lo;
        }
    }

    // ===================================================================
    // FROZEN MODEL CONSTANTS -- generated by v02/gen_data.py from
    //   research/nt8_port/csharp/harness_data/_model.json
    //   research/nt8_port/csharp/harness_data/_tmpl0.json
    // DO NOT hand-edit. Numbers are shortest-round-trip decimals (bit-exact).
    // ===================================================================
    static class ModelData
    {
        public static readonly string[] Topk = { "RSI06", "MACD07", "EXITKMDR", "TMPL0", "ZIGZAG", "ATR09", "NMP", "DOW19", "NMP9RIDEAGAINST", "ROUND05", "NMPTFADECALM", "RENKO24", "ORB02", "VWAP03", "CTXER", "PIVOT16", "SAR23", "PTRNENGULF", "NMP9RIDECALM", "NMPTMTFBRK", "TUNNEL20", "NMP9FADEAGAINST" };
        public static readonly string[] Cols = { "pivot_age_min", "sig_with_leg", "tod", "inter", "consensus", "is_RSI06", "is_MACD07", "is_EXITKMDR", "is_TMPL0", "is_ZIGZAG", "is_ATR09", "is_NMP", "is_DOW19", "is_NMP9RIDEAGAINST", "is_ROUND05", "is_NMPTFADECALM", "is_RENKO24", "is_ORB02", "is_VWAP03", "is_CTXER", "is_PIVOT16", "is_SAR23", "is_PTRNENGULF", "is_NMP9RIDECALM", "is_NMPTMTFBRK", "is_TUNNEL20", "is_NMP9FADEAGAINST" };
        public static readonly double[] Coef = {
            0.0975, 0.4304, 0.087, -0.1402, 0.2633, -0.3643, -0.2938, -0.2595, 0.2519, 0.1929,
            -0.1638, -0.1296, -0.0999, 0.0843, 0.0788, -0.0783, 0.0762, 0.0726, -0.072, -0.0604,
            -0.0585, -0.0581, 0.056, 0.051, 0.0495, 0.0466, 0.0466
        };
        public static readonly double[] Mu = {
            52.1754, 0.5084, 0.4515, 26.6785, 13.6748, 0.0154, 0.0101, 0.0224, 0.1601, 0.0049,
            0.0009, 0.0108, 0.0375, 0.0043, 0.033, 0.0212, 0.1626, 0.0006, 0.0318, 0.0239,
            0.0003, 0.0382, 0.0317, 0.0021, 0.0022, 0.0363, 0.0016
        };
        public static readonly double[] Sd = {
            38.3792, 0.4999, 0.2958, 38.0839, 5.4638, 0.1232, 0.1002, 0.1481, 0.3667, 0.0699,
            0.0303, 0.1034, 0.1901, 0.0653, 0.1785, 0.144, 0.369, 0.0236, 0.1755, 0.1528,
            0.0177, 0.1916, 0.1753, 0.0457, 0.047, 0.187, 0.0399
        };
        public const double Threshold = 0.7139834155227371;
        public const long ConsensusS = 180;
    }

    static class Tmpl0Data
    {
        public const int NTemplates = 1020;
        public static readonly double[] ScalerMean = { 0.950088, 2.743319, 6.500897, 0.257025, 0.713945, -0.035704 };
        public static readonly double[] ScalerScale = { 0.659283, 1.220591, 1.174041, 0.106095, 0.049625, 0.127959 };
        // 1020 templates x 6 dims, row-major (raw centroids, standardized at init)
        public static readonly double[] Centroids = {
            0.82499, 3.81084, 8.22882, 0.23278, 0.74228, -0.05905,
            0.81934, 1.47273, 5.90689, 0.19082, 0.74777, -0.01921,
            1.0663, 1.44725, 5.90689, 0.25631, 0.64833, -0.01637,
            0.41148, 3.84384, 5.90689, 0.35821, 0.76597, -0.1926,
            0.19593, 2.04397, 5.90689, 0.32641, 0.71196, 0.11724,
            1.45501, 3.72173, 5.90689, 0.44669, 0.75849, 0.12815,
            3.1673, 4.74987, 5.90689, 0.19506, 0.66116, -0.27229,
            0.876, 3.61538, 5.90689, 0.19125, 0.69238, -0.06719,
            2.10777, 4.15599, 5.90689, 0.15292, 0.72099, -0.02501,
            0.88703, 2.67646, 5.90689, 0.43651, 0.76062, -0.09075,
            1.42115, 4.00959, 8.22882, 0.4391, 0.76504, -0.17832,
            1.3444, 3.14312, 5.90689, 0.16959, 0.56751, 0.07083,
            0.22672, 2.73846, 5.90689, 0.19397, 0.6581, 0.06167,
            1.73955, 4.57508, 8.22882, 0.188, 0.63553, 0.07278,
            1.29748, 3.06238, 5.90689, 0.37262, 0.70823, -0.30393,
            0.24908, 3.45243, 8.22882, 0.21316, 0.75322, 0.05523,
            0.72824, 1.8642, 5.90689, 0.30688, 0.75865, -0.13261,
            0.3429, 0.24882, 5.90689, 0.2002, 0.66802, -0.09132,
            1.34856, 0.41851, 8.30613, 0.3768, 0.75778, 0.10924,
            0.52756, 3.09287, 5.90689, 0.21992, 0.75415, 0.09275,
            1.37659, 3.74139, 5.90689, 0.2127, 0.75208, -0.05407,
            1.43473, 1.13497, 5.90689, 0.36422, 0.75653, 0.08194,
            0.32955, 3.89843, 8.22882, 0.16077, 0.63581, -0.03696,
            0.39135, 2.32175, 5.90689, 0.55109, 0.75243, -0.2626,
            0.96697, 2.0588, 5.90689, 0.18331, 0.64796, -0.11289,
            2.52248, 3.13834, 5.90689, 0.35357, 0.75587, 0.05785,
            0.23142, 2.39191, 8.22882, 0.29488, 0.73613, -0.13746,
            0.84812, -0.0, 5.90689, 0.23497, 0.75864, 0.07963,
            1.24711, 3.21551, 5.90689, 0.19373, 0.69262, -0.13942,
            0.61518, 3.09713, 5.90689, 0.27377, 0.73137, -0.13402,
            0.33668, 1.22631, 8.22882, 0.23845, 0.7484, 0.05296,
            0.90568, 4.22664, 9.81378, 0.48987, 0.75805, -0.2368,
            0.98242, 2.98183, 9.81378, 0.1919, 0.73385, -0.03086,
            0.93431, 3.28996, 8.22882, 0.40803, 0.75608, 0.12926,
            1.18545, 3.85998, 5.90689, 0.14477, 0.62682, -0.01297,
            1.69177, 3.87964, 8.22882, 0.27741, 0.74696, -0.07662,
            0.171, 3.37915, 5.90689, 0.1729, 0.70375, -0.03989,
            0.28834, 2.35105, 5.90689, 0.12722, 0.53273, 0.01036,
            0.79621, 2.57253, 5.90689, 0.14774, 0.69462, 0.04163,
            2.04268, 4.39427, 9.81378, 0.32709, 0.75516, 0.0188,
            2.2795, 4.67378, 5.90689, 0.3613, 0.71798, -0.21702,
            1.87457, 3.38505, 5.90689, 0.31031, 0.76096, 0.07247,
            1.73867, 0.47871, 5.90689, 0.20159, 0.70457, -0.02888,
            1.92917, 2.68368, 5.90689, 0.19772, 0.6127, -0.14585,
            1.65286, 4.15197, 9.81378, 0.2024, 0.67665, -0.12326,
            0.67313, 1.68314, 5.90689, 0.14989, 0.70883, 0.02233,
            0.6998, 2.40184, 5.90689, 0.26072, 0.76356, 0.08134,
            0.25873, 1.51097, 8.22882, 0.1589, 0.65783, 0.04799,
            0.79318, 1.50911, 5.90689, 0.14827, 0.61983, 0.03898,
            0.8597, 2.88061, 5.90689, 0.45012, 0.76058, 0.22658,
            0.30937, 1.49082, 5.90689, 0.44623, 0.71181, -0.19207,
            2.66351, 3.62229, 5.90689, 0.22292, 0.65639, -0.22817,
            0.49937, 1.0219, 5.90689, 0.19733, 0.74093, -0.0879,
            0.26429, 1.68569, 5.90689, 0.45879, 0.7392, 0.22397,
            0.9097, 2.24657, 8.22882, 0.2624, 0.76023, 0.08135,
            2.09139, 3.10089, 5.90689, 0.14535, 0.72754, -0.093,
            0.90607, 3.29542, 9.81378, 0.26069, 0.73751, 0.10008,
            1.22469, 2.94287, 5.90689, 0.24605, 0.72917, 0.13052,
            2.47442, 5.14641, 9.81378, 0.36042, 0.6417, -0.33992,
            0.89054, 2.74709, 8.22882, 0.46214, 0.75423, -0.26066,
            0.28123, 4.72743, 9.81378, 0.2884, 0.73027, -0.09078,
            1.22906, 4.11903, 8.22882, 0.16906, 0.52833, -0.04665,
            1.45268, 3.00191, 5.90689, 0.35669, 0.74936, -0.21102,
            0.25602, 2.01357, 5.90689, 0.19893, 0.69606, -0.04669,
            1.40765, 1.18264, 5.90689, 0.19654, 0.72663, 0.00712,
            0.82317, 3.4644, 5.90689, 0.35439, 0.75744, 0.17106,
            2.12341, 3.79408, 5.90689, 0.22082, 0.5752, -0.1716,
            1.7218, 1.69506, 5.90689, 0.24573, 0.71367, -0.228,
            0.53304, 3.89487, 8.22882, 0.64076, 0.75359, -0.27132,
            0.53028, 4.27012, 8.22882, 0.42768, 0.75695, -0.27746,
            1.44239, 2.82325, 5.90689, 0.18992, 0.75601, 0.03461,
            3.16069, 5.38971, 8.22882, 0.22398, 0.64859, -0.2455,
            0.80278, 2.82217, 8.38731, 0.63275, 0.74678, 0.37034,
            1.5919, 3.30398, 5.90689, 0.53301, 0.76396, -0.25476,
            0.58834, 4.05507, 8.22882, 0.14623, 0.68252, 0.00207,
            1.44342, 1.22578, 8.22882, 0.19661, 0.68394, -0.11966,
            0.25796, 2.66364, 5.90689, 0.4057, 0.72409, -0.14642,
            0.64469, 0.91235, 5.90689, 0.19628, 0.72823, 0.04192,
            1.01844, 1.48198, 5.90689, 0.26747, 0.66399, 0.1505,
            0.98503, 2.04982, 5.90689, 0.14075, 0.56422, -0.04913,
            0.42412, 3.06105, 8.22882, 0.18886, 0.64763, 0.06147,
            0.40663, 3.43389, 5.90689, 0.26204, 0.73771, -0.04239,
            0.5154, 4.09709, 5.90689, 0.23167, 0.66001, -0.14738,
            0.83762, 1.09432, 5.90689, 0.27843, 0.73476, -0.15588,
            1.564, 3.55854, 5.90689, 0.12813, 0.68623, -0.01644,
            2.32056, 4.34865, 5.90689, 0.22292, 0.74645, -0.16064,
            0.92887, 0.68956, 5.90689, 0.47577, 0.75637, -0.15807,
            1.07252, 0.0, 5.90689, 0.13304, 0.68556, -0.02669,
            1.26305, 3.2349, 8.22882, 0.30555, 0.65303, -0.12668,
            1.51623, 3.62269, 8.22882, 0.13917, 0.72053, -0.03606,
            2.24876, 4.00854, 5.90689, 0.14195, 0.64465, -0.08492,
            2.1565, 4.28011, 8.22882, 0.39576, 0.74981, 0.13132,
            0.61601, 3.71015, 5.90689, 0.18175, 0.75166, -0.06425,
            1.16932, 1.16729, 5.90689, 0.44779, 0.76367, 0.21991,
            0.18641, 3.11342, 5.90689, 0.24651, 0.75341, -0.13285,
            0.42583, 2.12769, 5.90689, 0.32991, 0.53677, -0.10132,
            0.78494, 4.42032, 8.22882, 0.36358, 0.7515, 0.10834,
            1.03358, 4.34172, 5.90689, 0.39138, 0.75227, -0.21151,
            0.79062, 2.62085, 5.90689, 0.25759, 0.75543, -0.11607,
            0.24566, 2.22277, 5.90689, 0.12064, 0.59944, -0.01278,
            0.64845, 4.12916, 5.90689, 0.50311, 0.76348, -0.2234,
            0.81034, 0.25205, 8.22882, 0.15361, 0.68723, -0.00299,
            2.04082, 4.94075, 8.22882, 0.29891, 0.73076, -0.22263,
            1.63879, 1.36777, 5.90689, 0.2756, 0.73161, -0.07603,
            0.27054, 4.94847, 8.22882, 0.23099, 0.74186, -0.05949,
            0.20153, 1.87511, 5.90689, 0.14838, 0.74282, 0.03724,
            1.21979, 3.61288, 5.90689, 0.23391, 0.693, 0.16239,
            0.35301, 2.84188, 9.81378, 0.19623, 0.67231, -0.05106,
            2.27306, 3.53217, 5.90689, 0.37721, 0.762, 0.1494,
            0.27661, 1.9638, 5.90689, 0.16134, 0.67776, 0.05144,
            0.34788, 0.99488, 5.90689, 0.23136, 0.61998, -0.05793,
            0.57265, 2.06828, 5.90689, 0.23177, 0.71854, -0.0941,
            0.93208, 1.77324, 5.90689, 0.16916, 0.72298, -0.07263,
            0.74655, 3.04373, 5.90689, 0.13953, 0.66871, -0.05934,
            1.6341, 3.22714, 5.90689, 0.2831, 0.68826, -0.29954,
            0.43662, 2.58201, 8.22882, 0.40676, 0.74615, 0.2286,
            1.66353, 2.7623, 5.90689, 0.63531, 0.76117, 0.24685,
            2.7739, 4.51096, 9.81378, 0.21867, 0.56485, -0.19869,
            2.25537, 4.98386, 9.81378, 0.23078, 0.59542, -0.16682,
            1.41612, 1.53962, 5.90689, 0.27212, 0.76434, 0.07295,
            0.41708, 4.19112, 9.81378, 0.51009, 0.74621, 0.30634,
            0.55673, 2.6788, 5.90689, 0.1827, 0.71194, 0.08122,
            2.03996, 3.66345, 5.90689, 0.29759, 0.67746, -0.20923,
            0.32754, 5.30809, 9.81378, 0.20681, 0.67501, 0.02471,
            0.99091, 2.02082, 8.22882, 0.16364, 0.72437, -0.05664,
            1.62364, 3.00299, 5.90689, 0.28702, 0.72559, 0.03887,
            0.80659, 1.51944, 5.90689, 0.3646, 0.7022, 0.17261,
            0.22926, 3.17007, 8.22882, 0.14594, 0.59644, -0.01815,
            2.07316, 3.8475, 5.90689, 0.16522, 0.52423, -0.10205,
            0.72212, 4.11641, 8.22882, 0.30361, 0.7155, -0.20043,
            0.20456, 3.56594, 5.90689, 0.25005, 0.7537, 0.09811,
            2.45945, 3.98103, 5.90689, 0.14513, 0.71836, -0.11744,
            2.58269, 4.97576, 8.22882, 0.23313, 0.74372, -0.02274,
            0.26967, 3.859, 8.22882, 0.27971, 0.71026, -0.10574,
            1.08402, 0.37715, 5.90689, 0.25555, 0.68432, -0.17827,
            1.71552, 4.19317, 8.22882, 0.15943, 0.68545, -0.05545,
            0.32185, 1.0733, 8.22882, 0.47635, 0.72401, -0.13303,
            0.47292, 2.06366, 9.81378, 0.21167, 0.71208, -0.06407,
            1.86296, 4.1766, 8.22882, 0.16678, 0.57225, -0.10386,
            1.54201, 4.50715, 9.81378, 0.40315, 0.76033, 0.16392,
            1.97109, 2.89488, 5.90689, 0.20552, 0.75069, -0.05224,
            1.29404, 2.15978, 5.90689, 0.44982, 0.75698, -0.2818,
            1.80567, 5.35842, 9.81378, 0.26144, 0.69077, -0.13936,
            2.09804, 3.74519, 5.90689, 0.50443, 0.71384, -0.34472,
            1.63218, 4.33496, 8.22882, 0.17224, 0.74341, -0.07397,
            1.27396, 3.1362, 5.90689, 0.12984, 0.5709, -0.06032,
            1.51312, 2.9333, 5.90689, 0.26943, 0.76204, -0.07963,
            0.3808, 0.94727, 5.90689, 0.29272, 0.73588, 0.04598,
            0.71301, 3.18969, 8.22882, 0.11732, 0.72417, -0.00922,
            1.17403, 3.54614, 5.90689, 0.26568, 0.7313, -0.14846,
            0.39917, 3.03668, 5.90689, 0.26046, 0.73339, 0.18247,
            0.32071, 4.03271, 5.90689, 0.16831, 0.60166, 0.04009,
            0.18979, 2.48365, 5.90689, 0.3518, 0.74865, 0.1595,
            0.72173, 2.8787, 5.90689, 0.34202, 0.73893, -0.11838,
            0.371, 3.75607, 5.90689, 0.37039, 0.62725, -0.15729,
            1.09647, 3.40601, 5.90689, 0.21346, 0.72786, 0.00639,
            1.04967, 5.15522, 9.81378, 0.17725, 0.60521, -0.01793,
            1.42942, 1.92933, 5.90689, 0.19689, 0.6801, 0.02146,
            0.27525, 0.8831, 5.90689, 0.19665, 0.69293, 0.09186,
            1.23635, 1.73866, 8.22882, 0.34083, 0.75382, -0.08042,
            1.92259, 3.46119, 5.90689, 0.44042, 0.76072, -0.21361,
            0.39767, 2.73441, 5.90689, 0.41565, 0.73776, -0.31224,
            0.67, 3.90773, 5.90689, 0.21306, 0.7259, -0.11163,
            1.01705, 4.43687, 9.81378, 0.2988, 0.69574, -0.20176,
            0.92486, 3.75513, 8.22882, 0.20719, 0.65435, 0.01252,
            1.04021, 1.98822, 5.90689, 0.42204, 0.76071, 0.09036,
            1.28005, 0.17913, 5.90689, 0.34788, 0.75677, -0.17973,
            0.2388, 1.4896, 5.90689, 0.34073, 0.72529, -0.17654,
            0.3742, 3.4208, 5.90689, 0.25995, 0.67183, -0.00634,
            0.6403, 2.6385, 5.90689, 0.19331, 0.74668, -0.04241,
            0.70224, 2.54359, 5.90689, 0.25587, 0.73447, 0.12077,
            1.57274, 2.26144, 5.90689, 0.34114, 0.7582, -0.09573,
            1.4313, 3.06833, 5.90689, 0.13733, 0.62847, -0.08873,
            0.23229, 3.13858, 9.81378, 0.3111, 0.74466, 0.0964,
            0.85613, 2.03569, 9.81378, 0.3504, 0.75589, -0.11419,
            0.52408, 1.0107, 5.90689, 0.25779, 0.72248, -0.14314,
            2.13044, 3.80489, 8.22882, 0.28624, 0.75713, 0.05868,
            1.13777, 4.09802, 8.22882, 0.37709, 0.75305, -0.08791,
            0.7347, 0.78849, 5.90689, 0.13662, 0.64416, -0.03465,
            0.74032, 3.55964, 8.22882, 0.18399, 0.72052, 0.0528,
            0.39049, 3.70945, 5.90689, 0.23263, 0.6278, -0.05135,
            0.59622, 3.37853, 5.90689, 0.24292, 0.71864, 0.07398,
            1.13552, 1.43745, 5.90689, 0.20481, 0.76031, 0.04878,
            1.07962, 2.21434, 5.90689, 0.38932, 0.76077, 0.19477,
            1.20826, 0.94149, 5.90689, 0.15828, 0.75258, -0.01586,
            0.72487, 2.08534, 9.81378, 0.20901, 0.63153, 0.06149,
            0.60168, 0.28159, 5.90689, 0.12821, 0.59064, -0.0048,
            1.67603, 3.30324, 5.90689, 0.32454, 0.64547, 0.21992,
            0.42205, -0.0, 5.90689, 0.14574, 0.69787, 0.00871,
            1.73452, 3.40771, 5.90689, 0.16395, 0.69165, -0.10242,
            0.25004, 0.95863, 8.28752, 0.24298, 0.67621, -0.07509,
            1.22894, 2.54087, 8.22882, 0.23904, 0.6806, -0.12517,
            0.21434, 0.70632, 5.90689, 0.28665, 0.74689, 0.16253,
            0.66401, 1.8167, 9.81378, 0.23156, 0.75263, 0.06229,
            0.74661, 1.06053, 5.90689, 0.41627, 0.74786, 0.14841,
            1.64216, 3.84704, 5.90689, 0.19167, 0.71684, -0.02206,
            0.31104, 2.99525, 5.90689, 0.40255, 0.76127, -0.17515,
            0.36698, 3.13081, 8.22882, 0.22407, 0.55018, -0.08369,
            0.21135, 4.04519, 9.81378, 0.19481, 0.73645, 0.00585,
            0.30848, 1.1091, 8.42694, 0.54578, 0.7534, -0.27405,
            0.34914, 4.60361, 9.81378, 0.45354, 0.75629, -0.17011,
            0.77152, 2.52554, 5.90689, 0.23517, 0.63297, 0.06995,
            1.26783, 4.38665, 9.81378, 0.17026, 0.73954, 0.01506,
            0.26826, 2.70282, 5.90689, 0.12654, 0.69018, -0.02091,
            1.26698, 4.00092, 5.90689, 0.1695, 0.69009, 0.04592,
            1.20367, 1.4015, 5.90689, 0.27032, 0.55845, 0.04977,
            1.01712, 4.13922, 8.22882, 0.47079, 0.72339, -0.36258,
            1.32344, 2.6837, 5.90689, 0.50177, 0.75669, 0.15851,
            0.74011, 2.5496, 5.90689, 0.37482, 0.7477, -0.21894,
            1.37078, 2.75404, 8.22882, 0.44739, 0.76531, 0.18813,
            1.42867, 0.30524, 5.90689, 0.29302, 0.76369, 0.09039,
            0.18726, 2.21456, 5.90689, 0.22079, 0.73456, -0.10749,
            0.16948, 0.90918, 5.90689, 0.12008, 0.66405, -0.00028,
            0.70979, 3.46373, 5.90689, 0.13591, 0.5988, -0.02558,
            0.91952, 3.8722, 5.90689, 0.59652, 0.74889, -0.2865,
            1.89276, 4.31604, 5.90689, 0.27832, 0.71237, -0.26225,
            0.92837, 3.65056, 5.90689, 0.20252, 0.63803, 0.10804,
            2.31774, 5.40676, 9.81378, 0.20864, 0.60476, 0.09451,
            1.52778, 2.96846, 8.22882, 0.31244, 0.73876, -0.12289,
            0.33732, 4.25504, 8.22882, 0.33802, 0.75799, -0.15708,
            0.34003, 4.43357, 5.90689, 0.32634, 0.75873, -0.13537,
            0.43496, 0.8217, 8.22882, 0.32487, 0.72831, -0.16058,
            1.20119, 3.89711, 5.90689, 0.24624, 0.75417, 0.07061,
            1.60249, 4.78612, 8.22882, 0.56684, 0.74807, -0.31968,
            1.46357, 2.69202, 8.22882, 0.15457, 0.68176, -0.06192,
            1.44971, 3.59719, 5.90689, 0.13135, 0.51584, 0.01257,
            0.22484, 3.30195, 8.22882, 0.13695, 0.72852, 0.00082,
            1.49981, 2.65721, 5.90689, 0.33274, 0.76686, 0.05691,
            1.22396, 2.91987, 5.90689, 0.13617, 0.74584, -0.00818,
            1.03721, 2.56749, 5.90689, 0.13473, 0.72303, -0.02906,
            0.28688, 3.43589, 5.90689, 0.13686, 0.74038, -0.01298,
            3.44995, 5.034, 5.90689, 0.18463, 0.53953, -0.35045,
            0.2734, 1.72778, 8.22882, 0.22448, 0.7112, 0.10652,
            1.19347, 4.26416, 8.22882, 0.23288, 0.75901, 0.00457,
            1.39484, 1.71407, 5.90689, 0.36752, 0.6995, -0.27167,
            0.65481, 0.34527, 8.30807, 0.15343, 0.62862, -0.0379,
            0.2613, 3.87487, 5.90689, 0.17217, 0.70267, 0.04294,
            0.83899, 2.25402, 5.90689, 0.44854, 0.76681, -0.17425,
            2.3418, 4.03058, 5.90689, 0.31916, 0.70619, -0.32221,
            1.56141, 2.57683, 5.90689, 0.4193, 0.76395, 0.20433,
            3.04846, 5.65734, 9.81378, 0.30777, 0.71741, -0.23319,
            0.26862, 4.80662, 9.81378, 0.46531, 0.75548, -0.29686,
            0.20167, 1.49331, 5.90689, 0.17489, 0.715, 0.07692,
            2.40217, 3.13413, 5.90689, 0.22994, 0.59061, -0.33599,
            0.31837, 2.86257, 8.22882, 0.31717, 0.74551, 0.14806,
            1.29301, 1.00002, 5.90689, 0.15609, 0.54316, 0.0757,
            0.18635, 0.97843, 5.90689, 0.1506, 0.73797, -0.02442,
            0.7893, 1.50634, 5.90689, 0.21234, 0.70001, -0.13788,
            0.40455, 2.68309, 5.90689, 0.31238, 0.76389, -0.15567,
            0.26851, 1.27137, 5.90689, 0.25099, 0.72292, 0.1179,
            1.00408, 3.50042, 5.90689, 0.44954, 0.73557, -0.12969,
            2.45738, 4.6056, 8.22882, 0.1642, 0.7365, -0.10016,
            0.4247, 1.8305, 5.90689, 0.34652, 0.68826, -0.11737,
            0.65306, 2.37385, 5.90689, 0.12643, 0.65319, -0.00454,
            1.54663, 3.42546, 5.90689, 0.25233, 0.7104, -0.07561,
            1.10258, 3.20661, 8.22882, 0.16382, 0.61729, -0.08848,
            0.21899, 1.19717, 5.90689, 0.13616, 0.61434, 0.02974,
            1.65463, 0.90246, 5.90689, 0.69147, 0.75143, -0.29472,
            1.7839, 2.34961, 5.90689, 0.13748, 0.69494, -0.07426,
            0.77229, 1.74702, 8.22882, 0.23229, 0.63498, -0.07697,
            2.25581, 4.33756, 5.90689, 0.32881, 0.72292, 0.0374,
            2.01417, 1.27035, 5.90689, 0.23991, 0.62798, 0.28486,
            1.57769, 0.66243, 5.90689, 0.22655, 0.59887, 0.17644,
            0.83392, 1.42956, 5.90689, 0.37692, 0.7515, -0.22371,
            2.67172, 3.71265, 5.90689, 0.17214, 0.74204, -0.06164,
            1.54698, 1.70379, 5.90689, 0.1733, 0.59573, -0.14329,
            1.33812, 1.24753, 5.90689, 0.23891, 0.76145, -0.02535,
            1.55162, 3.31328, 8.22882, 0.581, 0.75844, -0.19129,
            0.63761, 2.40402, 5.90689, 0.25095, 0.73262, -0.00188,
            0.73218, 4.53295, 9.81378, 0.24399, 0.76033, -0.07903,
            2.92208, 4.3717, 5.90689, 0.37201, 0.75016, 0.04366,
            0.43527, 3.13261, 8.27285, 0.4699, 0.69764, 0.13197,
            0.84712, 3.33741, 5.90689, 0.19484, 0.76082, 0.02797,
            1.12897, 4.81256, 8.22882, 0.31984, 0.71623, -0.05724,
            0.52641, 3.21567, 5.90689, 0.5432, 0.74557, 0.37395,
            1.16242, 3.90921, 5.90689, 0.45005, 0.67972, -0.23984,
            2.60558, 5.15566, 8.22882, 0.37232, 0.69188, -0.37634,
            1.65628, 1.41104, 5.90689, 0.51078, 0.7596, -0.16253,
            1.69712, 1.85507, 5.90689, 0.3291, 0.63417, -0.21767,
            0.71159, 0.28253, 5.90689, 0.28415, 0.72892, 0.2108,
            0.54076, 0.96456, 9.81378, 0.30926, 0.73607, -0.11293,
            2.21839, 4.00635, 5.90689, 0.18062, 0.62237, 0.05021,
            1.18204, 2.80947, 5.90689, 0.26975, 0.71342, -0.21553,
            1.66852, 1.92523, 8.22882, 0.29816, 0.66336, -0.31735,
            0.40127, 1.13913, 8.40996, 0.53192, 0.75299, 0.26033,
            0.85784, 1.77161, 5.90689, 0.22278, 0.74768, 0.0762,
            0.33464, 3.54338, 5.90689, 0.32527, 0.73075, -0.22375,
            1.33999, 4.9309, 9.81378, 0.50911, 0.72844, -0.38605,
            0.26168, 3.46144, 5.90689, 0.39435, 0.7535, -0.26282,
            1.91215, 3.64714, 5.90689, 0.58102, 0.76697, 0.07568,
            2.28753, 3.58339, 5.90689, 0.4962, 0.76429, 0.11516,
            2.84307, 3.68228, 5.90689, 0.55372, 0.75514, 0.08326,
            2.0892, 4.78208, 8.22882, 0.40055, 0.76048, -0.07607,
            1.87005, 5.35037, 9.81378, 0.3949, 0.71242, -0.26305,
            1.46683, 3.15775, 8.22882, 0.20233, 0.62861, 0.11214,
            0.92065, 2.02253, 8.22882, 0.1399, 0.57984, -0.02301,
            0.98069, 3.38399, 5.90689, 0.43871, 0.72159, 0.15559,
            0.87719, 0.94607, 5.90689, 0.19968, 0.72206, -0.06107,
            0.2042, 2.80156, 5.90689, 0.20143, 0.60474, -0.04809,
            2.43369, 4.50935, 8.22882, 0.21842, 0.64541, -0.23318,
            0.67128, 1.90821, 5.90689, 0.15751, 0.68449, -0.09086,
            1.55561, 2.80963, 5.90689, 0.13602, 0.7128, -0.0202,
            1.78513, 3.91799, 5.90689, 0.19432, 0.75265, 0.03158,
            0.27739, 1.74783, 9.81378, 0.33152, 0.73023, 0.1767,
            0.25076, 3.02308, 5.90689, 0.28427, 0.69445, 0.11991,
            1.57372, 3.84766, 5.90689, 0.16684, 0.64956, -0.09168,
            0.4206, 3.80228, 5.90689, 0.44388, 0.75393, 0.23795,
            0.65223, 3.3357, 5.90689, 0.12488, 0.69647, 0.00944,
            1.33752, 4.59152, 5.90689, 0.25323, 0.72103, 0.04844,
            1.03706, 3.10189, 5.90689, 0.25316, 0.67364, -0.17591,
            1.33114, 3.67589, 5.90689, 0.25228, 0.66499, -0.03109,
            0.27035, 0.5336, 5.90689, 0.376, 0.7495, -0.23363,
            1.32758, 0.79305, 5.90689, 0.15002, 0.67318, 0.04427,
            2.13927, 4.07897, 9.81378, 0.51657, 0.76011, 0.14124,
            0.54493, 2.45722, 5.90689, 0.12716, 0.71869, -0.00832,
            1.18875, 0.022, 5.90689, 0.4081, 0.75649, 0.16124,
            1.62422, 4.58127, 8.22882, 0.25468, 0.72239, 0.16972,
            1.20827, 1.8788, 5.90689, 0.34734, 0.75347, -0.15074,
            0.14845, 2.20169, 8.22882, 0.19464, 0.68828, -0.0177,
            2.00109, 3.50913, 5.90689, 0.28572, 0.75858, -0.07073,
            2.32389, 4.63994, 8.22882, 0.17726, 0.68213, -0.13769,
            0.37452, 3.39326, 9.81378, 0.15376, 0.61288, -0.00397,
            1.74298, 2.43575, 5.90689, 0.15155, 0.74489, -0.01615,
            0.1941, 2.02707, 5.90689, 0.21153, 0.7417, 0.10806,
            1.08396, 3.58421, 9.81378, 0.49025, 0.76533, 0.17663,
            0.24642, -0.0, 5.90689, 0.20265, 0.72954, -0.03578,
            0.77543, 4.08379, 9.81378, 0.22078, 0.75239, 0.04442,
            1.10315, 3.11198, 5.90689, 0.23751, 0.75701, -0.0695,
            0.38504, 2.13597, 5.90689, 0.36524, 0.74477, 0.23095,
            1.64891, 4.57171, 5.90689, 0.55514, 0.74088, -0.37493,
            0.21347, 3.36002, 5.90689, 0.29566, 0.71804, -0.08808,
            0.71523, 1.81934, 5.90689, 0.18243, 0.62364, -0.05489,
            0.60587, 1.48623, 5.90689, 0.13246, 0.58212, -0.00372,
            0.84762, 3.87473, 5.90689, 0.48905, 0.75015, -0.34246,
            0.53484, 2.49546, 5.90689, 0.1869, 0.65678, -0.08845,
            0.19672, 3.15819, 5.90689, 0.26993, 0.72409, -0.17035,
            2.9538, 4.88756, 5.90689, 0.19592, 0.66085, 0.12488,
            2.54702, 4.7843, 5.90689, 0.25305, 0.63764, 0.22118,
            0.24965, 2.7886, 8.22882, 0.44497, 0.7529, -0.19067,
            1.55187, 3.95442, 8.22882, 0.24011, 0.69117, -0.19446,
            1.32396, 1.91191, 5.90689, 0.27645, 0.7596, -0.09919,
            1.22018, 1.02368, 8.22882, 0.23337, 0.70733, 0.18022,
            1.24475, 1.78761, 5.90689, 0.26687, 0.70431, -0.07482,
            0.38242, 4.3757, 5.90689, 0.37499, 0.73918, -0.27269,
            1.2762, 3.4941, 5.90689, 0.33909, 0.58648, -0.18628,
            1.39901, 3.54454, 5.90689, 0.14357, 0.71958, -0.07154,
            1.44521, 1.5857, 8.22882, 0.23046, 0.73315, -0.10979,
            0.73736, 4.00639, 8.22882, 0.28402, 0.61932, 0.01846,
            0.82847, 3.23825, 5.90689, 0.3686, 0.76645, -0.15925,
            0.24071, 4.83862, 8.22882, 0.29498, 0.72734, -0.16423,
            1.69337, 2.80334, 9.81378, 0.30088, 0.74097, 0.03688,
            0.60882, 0.33887, 5.90689, 0.39018, 0.73931, -0.15615,
            1.70684, 1.81458, 8.22882, 0.26517, 0.7516, 0.05852,
            1.06697, 3.62251, 5.90689, 0.13947, 0.71826, -0.00755,
            0.26584, 2.0261, 5.90689, 0.25431, 0.66319, -0.06875,
            1.52741, 0.86983, 5.90689, 0.24614, 0.698, 0.19375,
            0.262, 3.64089, 8.22882, 0.44884, 0.75253, 0.17619,
            0.23676, 2.30213, 5.90689, 0.15193, 0.66889, -0.04984,
            0.39845, 0.98539, 5.90689, 0.37683, 0.74928, 0.24496,
            3.27799, 4.82325, 5.90689, 0.31451, 0.72384, -0.21788,
            1.49799, 2.62423, 8.22882, 0.32266, 0.75624, 0.0925,
            0.796, 1.58932, 9.81378, 0.49171, 0.76494, -0.19308,
            1.39097, 0.97145, 9.81378, 0.50153, 0.76583, -0.19931,
            1.59777, 2.83183, 5.90689, 0.21252, 0.72629, -0.02154,
            0.20524, 3.06997, 8.22882, 0.21365, 0.72687, -0.04832,
            0.6924, 1.05386, 8.22882, 0.16693, 0.73089, -0.00147,
            1.1415, 4.53694, 5.90689, 0.24682, 0.73775, -0.14867,
            0.99914, 0.97023, 5.90689, 0.33808, 0.7626, -0.1209,
            2.44317, 4.31997, 8.22882, 0.32294, 0.7361, -0.08139,
            0.27511, 1.96343, 8.22882, 0.14931, 0.73702, 0.01853,
            0.20762, 3.17637, 5.90689, 0.12688, 0.71065, 0.01217,
            1.43778, 3.5717, 5.90689, 0.18711, 0.72998, 0.06061,
            0.86103, 3.05284, 5.90689, 0.30302, 0.65615, 0.11914,
            0.93378, 0.89253, 5.90689, 0.13283, 0.70993, -0.01857,
            0.5828, 4.28547, 9.81378, 0.24952, 0.62575, -0.10332,
            1.67317, 0.58061, 5.90689, 0.4095, 0.76553, 0.18372,
            1.11109, 2.51442, 5.90689, 0.26652, 0.63505, -0.11184,
            1.29498, 3.1858, 8.22882, 0.23605, 0.75455, 0.05378,
            0.87341, 2.79091, 5.90689, 0.35073, 0.72122, 0.09266,
            0.99753, 3.53935, 8.22882, 0.27432, 0.69948, -0.00352,
            0.28361, 3.32104, 8.22882, 0.2024, 0.70509, 0.08353,
            1.70137, 1.73654, 5.90689, 0.19223, 0.65937, -0.14853,
            2.59237, 5.40769, 9.81378, 0.22927, 0.64522, -0.23776,
            3.31071, 5.77019, 9.81378, 0.2021, 0.64879, -0.22805,
            0.39975, 2.46446, 5.90689, 0.44633, 0.75086, 0.30327,
            0.64426, 1.79852, 5.90689, 0.19447, 0.72006, 0.08248,
            1.20846, 3.95758, 5.90689, 0.32887, 0.72304, 0.12551,
            2.30802, 3.66257, 5.90689, 0.19907, 0.70777, -0.19577,
            1.04235, 0.39989, 8.22882, 0.3148, 0.72368, -0.01823,
            1.70114, 3.33608, 5.90689, 0.14641, 0.74369, -0.01702,
            0.3249, 3.70435, 5.90689, 0.47016, 0.76005, -0.25732,
            2.03565, 3.48555, 5.90689, 0.18366, 0.64452, -0.17632,
            1.00077, 2.50489, 5.90689, 0.20463, 0.72446, -0.15606,
            1.25394, 3.56391, 9.81378, 0.56728, 0.76796, -0.24803,
            2.23322, 4.36402, 9.81378, 0.20294, 0.73098, -0.01324,
            0.32739, 2.60188, 5.90689, 0.31947, 0.70268, -0.13283,
            1.28929, 0.76575, 5.90689, 0.15226, 0.57463, -0.08453,
            0.26158, 3.30202, 8.22882, 0.32, 0.7476, -0.20653,
            0.18601, 3.92151, 5.90689, 0.1896, 0.74135, -0.01379,
            0.74558, 4.61242, 9.81378, 0.37225, 0.73602, 0.1182,
            2.00764, 4.49014, 5.90689, 0.19832, 0.68179, -0.15388,
            0.60802, 1.6884, 8.22882, 0.13834, 0.68836, -0.01582,
            1.54437, 0.94789, 5.90689, 0.53204, 0.76112, 0.13329,
            2.1643, 3.75497, 5.90689, 0.40298, 0.75318, 0.03785,
            0.27937, 3.35867, 9.81378, 0.26631, 0.73779, -0.08212,
            0.27042, 2.12143, 5.90689, 0.15056, 0.72282, -0.06341,
            0.71463, 3.76654, 8.22882, 0.13403, 0.56147, 0.01611,
            0.20052, 3.4353, 5.90689, 0.13589, 0.67098, -0.01047,
            0.30282, 2.7564, 8.22882, 0.30359, 0.68675, -0.12508,
            0.42085, 3.25175, 5.90689, 0.27509, 0.5834, 0.07704,
            0.84506, 2.05065, 5.90689, 0.1273, 0.68472, -0.02253,
            0.18512, 4.05971, 9.81378, 0.16866, 0.68705, -0.00848,
            0.89154, 1.86236, 5.90689, 0.18282, 0.6693, 0.08905,
            0.3659, 2.54344, 5.90689, 0.25507, 0.61196, -0.11078,
            0.3794, 2.31099, 8.22882, 0.18081, 0.72695, -0.08245,
            0.64065, 4.90764, 8.22882, 0.24615, 0.73559, 0.05352,
            0.75869, 4.47318, 9.81378, 0.17505, 0.70829, 0.01338,
            1.08733, 2.10941, 5.90689, 0.55257, 0.76442, 0.24104,
            1.43523, 0.95491, 5.90689, 0.21186, 0.74833, -0.17103,
            0.38022, 4.2926, 8.22882, 0.18335, 0.58178, -0.0501,
            0.17697, 1.44379, 5.90689, 0.22242, 0.75012, -0.08891,
            0.90923, 3.80861, 5.90689, 0.14668, 0.67172, -0.00405,
            1.63147, 3.88124, 5.90689, 0.27294, 0.7499, -0.06427,
            0.25622, 1.81601, 5.90689, 0.41551, 0.74954, -0.28016,
            2.46946, 4.39038, 5.90689, 0.29897, 0.60834, -0.25053,
            0.93153, 2.89648, 5.90689, 0.36603, 0.70416, -0.17304,
            0.43909, 4.1839, 5.90689, 0.29257, 0.71261, -0.09827,
            0.39062, 3.41501, 5.90689, 0.69229, 0.7462, -0.26282,
            0.4053, 1.02697, 9.81378, 0.1729, 0.73169, -0.00368,
            2.28223, 3.5718, 5.90689, 0.27591, 0.71229, -0.04578,
            0.3199, 0.32675, 5.90689, 0.15197, 0.54679, -0.00018,
            0.81691, 1.0483, 5.90689, 0.23037, 0.70316, 0.14763,
            1.77844, 4.24589, 5.90689, 0.36465, 0.73293, -0.22258,
            0.50178, 4.44425, 8.22882, 0.23374, 0.67047, -0.0965,
            1.87669, 4.24307, 9.81378, 0.2869, 0.7209, -0.21611,
            2.14888, 4.03088, 8.22882, 0.53428, 0.76136, 0.1651,
            1.76412, 3.82201, 5.90689, 0.35394, 0.73331, -0.32649,
            0.42984, 2.0105, 8.22882, 0.34273, 0.6344, -0.13025,
            0.3778, 3.71342, 8.22882, 0.43012, 0.72043, -0.19235,
            1.0694, 2.99235, 8.22882, 0.2637, 0.75753, -0.06916,
            0.74645, 1.04447, 5.90689, 0.13598, 0.52259, -0.03146,
            0.32166, 5.07435, 8.22882, 0.44022, 0.75766, -0.24156,
            1.23824, 3.65949, 5.90689, 0.40328, 0.75251, 0.22595,
            0.4053, 1.55261, 5.90689, 0.2545, 0.75764, 0.09637,
            0.26446, 2.19285, 5.90689, 0.22712, 0.70682, 0.12267,
            0.76642, 0.88811, 5.90689, 0.12912, 0.74386, -0.00178,
            1.91787, 2.92032, 5.90689, 0.20131, 0.69533, -0.17721,
            0.85004, 3.14987, 8.22882, 0.16352, 0.69038, -0.04901,
            0.29415, 2.92549, 5.90689, 0.39783, 0.69618, 0.12039,
            2.78, 4.9583, 8.22882, 0.38527, 0.75295, 0.03093,
            1.28584, 3.06082, 5.90689, 0.33629, 0.75977, 0.14368,
            0.90002, 0.888, 5.90689, 0.25463, 0.74479, -0.04668,
            1.4772, 3.22953, 5.90689, 0.42042, 0.7657, -0.13176,
            1.61046, 3.70352, 5.90689, 0.24835, 0.74099, 0.13959,
            1.42143, 4.10435, 8.22882, 0.32287, 0.74176, -0.23879,
            0.241, 3.50182, 5.90689, 0.19816, 0.73254, -0.09572,
            0.99281, 4.32158, 8.22882, 0.52088, 0.75212, -0.19862,
            0.99834, 3.1545, 5.90689, 0.29615, 0.75714, 0.09235,
            0.59651, 2.99069, 5.90689, 0.40886, 0.75673, 0.11788,
            0.97538, 4.1744, 9.81378, 0.22591, 0.71457, -0.08343,
            0.96914, 0.12691, 5.90689, 0.2042, 0.68563, 0.09063,
            0.98616, 2.59101, 5.90689, 0.32789, 0.76914, -0.12068,
            0.51794, 2.43839, 5.90689, 0.18517, 0.75414, 0.06077,
            2.07871, 3.6423, 5.90689, 0.48169, 0.76199, 0.25676,
            0.19431, 3.02202, 5.90689, 0.1855, 0.69705, 0.05771,
            0.83988, 3.8267, 8.22882, 0.26138, 0.74956, 0.10657,
            0.206, 0.98345, 5.90689, 0.3101, 0.65426, -0.12085,
            0.25361, 2.90018, 9.81378, 0.37469, 0.74201, -0.14939,
            1.96252, 3.65219, 5.90689, 0.14567, 0.60576, -0.0936,
            1.89518, 4.05283, 9.81378, 0.3426, 0.69949, -0.0555,
            1.30894, 1.75865, 9.81378, 0.20447, 0.74181, -0.02764,
            0.70925, 1.06206, 5.90689, 0.22858, 0.68755, -0.03892,
            1.50669, 3.00936, 5.90689, 0.21169, 0.65078, -0.14524,
            2.57488, 4.25051, 5.90689, 0.31632, 0.75505, -0.08558,
            1.33624, 2.06973, 5.90689, 0.20298, 0.73665, 0.06017,
            2.70015, 4.28616, 5.90689, 0.18688, 0.52231, -0.18484,
            2.64922, 5.12393, 8.22882, 0.23362, 0.71329, -0.24449,
            0.30639, 0.82453, 5.90689, 0.31423, 0.70673, 0.15535,
            2.18807, 5.05986, 9.73453, 0.33244, 0.70333, 0.27279,
            0.79686, -0.0, 5.90689, 0.15064, 0.74248, -0.0208,
            0.61157, 4.38083, 9.81378, 0.42572, 0.71343, -0.13687,
            3.15229, 5.65612, 9.39112, 0.24881, 0.58716, -0.40836,
            1.82272, 3.51819, 8.22882, 0.40488, 0.69216, -0.32989,
            3.34492, 5.92428, 9.16539, 0.43332, 0.69918, -0.3729,
            1.1751, 0.23299, 5.90689, 0.28457, 0.75421, -0.09669,
            0.21156, 1.12952, 5.90689, 0.18395, 0.67226, -0.00924,
            0.31021, 0.87241, 5.90689, 0.50198, 0.75198, -0.27725,
            1.93681, 4.50043, 5.90689, 0.41788, 0.70374, -0.32107,
            0.62807, 3.13051, 5.90689, 0.13956, 0.64401, 0.02966,
            0.90387, 1.98923, 8.22882, 0.1871, 0.74293, 0.04267,
            0.81231, 3.1143, 5.90689, 0.2952, 0.6763, -0.07545,
            2.61799, 4.30169, 5.90689, 0.39598, 0.73061, -0.33349,
            1.70612, 2.77701, 5.90689, 0.25363, 0.74485, -0.15628,
            1.46087, 1.81777, 5.90689, 0.16002, 0.75882, 0.00149,
            2.25127, 3.53107, 5.90689, 0.14889, 0.68532, -0.08694,
            0.30358, 2.3621, 5.90689, 0.252, 0.75343, -0.14753,
            1.24156, 2.33695, 5.90689, 0.32979, 0.7449, 0.02031,
            0.961, 3.40665, 9.81378, 0.32041, 0.74273, -0.10189,
            0.35416, 3.42363, 8.22882, 0.25486, 0.63349, -0.09944,
            0.19143, 2.80053, 5.90689, 0.18011, 0.72799, -0.03178,
            2.05037, 3.92188, 5.90689, 0.37235, 0.75536, -0.09749,
            0.81643, 3.93895, 5.90689, 0.27224, 0.69701, -0.22369,
            1.99645, 2.29639, 5.90689, 0.39126, 0.75774, -0.15046,
            1.63863, 2.15287, 5.90689, 0.23939, 0.75179, -0.01627,
            1.90492, 4.08515, 8.22882, 0.17783, 0.74597, 0.00479,
            0.58446, 2.0518, 5.90689, 0.28568, 0.72231, 0.18506,
            1.18326, 4.12163, 8.22882, 0.24503, 0.68202, -0.10087,
            0.49296, 4.0378, 8.22882, 0.55512, 0.75962, 0.31289,
            0.85856, 4.31941, 8.22882, 0.40976, 0.74507, 0.24226,
            0.42969, 0.91516, 5.90689, 0.13503, 0.72016, 0.03998,
            2.64697, 5.31995, 9.81378, 0.23787, 0.7393, -0.08119,
            1.34222, 5.10531, 9.81378, 0.30555, 0.7321, -0.1121,
            0.20491, 2.57761, 5.90689, 0.25847, 0.69557, -0.02154,
            0.78656, 2.40684, 5.90689, 0.31317, 0.73272, -0.21895,
            1.07913, 0.99736, 8.22882, 0.23667, 0.75028, 0.06422,
            1.05146, 3.05831, 5.90689, 0.14835, 0.69984, -0.05601,
            0.30583, 1.37057, 8.22882, 0.24074, 0.73647, -0.10272,
            0.27734, 3.45749, 8.22882, 0.21726, 0.68694, -0.05509,
            1.24418, 0.92965, 5.90689, 0.24467, 0.74631, 0.10793,
            0.18203, 2.99926, 5.90689, 0.20943, 0.75093, 0.04636,
            1.54058, 0.88637, 5.90689, 0.18262, 0.64335, 0.11976,
            2.17955, 5.12938, 8.22882, 0.19873, 0.58794, -0.13585,
            1.78177, 4.81476, 9.81378, 0.55099, 0.76502, -0.21921,
            1.88188, 1.52394, 5.90689, 0.3016, 0.75592, 0.08389,
            0.1555, 3.19516, 5.90689, 0.17447, 0.72988, 0.0769,
            0.59867, 2.74462, 5.90689, 0.30187, 0.75286, 0.15991,
            1.45493, 1.79367, 5.90689, 0.17305, 0.7249, -0.05347,
            2.9726, 5.18662, 8.22882, 0.18247, 0.54128, -0.19485,
            0.70427, 3.09506, 5.90689, 0.13092, 0.74093, 0.00042,
            0.94351, 2.91242, 5.90689, 0.16228, 0.61389, -0.11212,
            0.32668, 4.05504, 8.22882, 0.25254, 0.75058, -0.12091,
            1.56059, 4.42843, 9.81378, 0.22866, 0.68184, 0.07609,
            0.7734, 1.95479, 5.90689, 0.28671, 0.70155, 0.07199,
            1.24054, 3.93173, 5.90689, 0.30217, 0.73491, -0.25992,
            1.6019, 0.91576, 5.90689, 0.2253, 0.75311, 0.03391,
            1.71012, 2.92963, 5.90689, 0.2616, 0.64673, 0.01387,
            2.19601, 3.15995, 5.90689, 0.17638, 0.71301, -0.01426,
            0.26093, 3.49627, 5.90689, 0.53544, 0.73527, -0.19638,
            1.17583, 2.31381, 8.28542, 0.25077, 0.56836, -0.11737,
            0.97684, 1.00708, 5.90689, 0.37699, 0.71611, -0.05947,
            1.4282, 1.99416, 9.81378, 0.26177, 0.69636, 0.2017,
            0.66706, 3.64293, 5.90689, 0.15348, 0.53385, -0.01129,
            1.76386, 0.9916, 5.90689, 0.31616, 0.7604, -0.09124,
            0.48168, 2.18217, 9.81378, 0.2358, 0.70023, 0.0894,
            1.33171, 0.49298, 8.26116, 0.28365, 0.75671, -0.08807,
            0.1987, 2.46449, 5.90689, 0.17473, 0.75088, -0.07237,
            0.96099, 4.08439, 5.90689, 0.26219, 0.74282, -0.03248,
            3.30372, 5.42917, 7.81491, 0.22149, 0.69751, 0.26822,
            0.26132, 1.76824, 5.90689, 0.24407, 0.68435, 0.05914,
            0.88714, 4.25423, 8.22882, 0.15442, 0.73324, -0.01746,
            0.47403, 0.60739, 9.81378, 0.33265, 0.74073, 0.16892,
            1.2812, 1.20298, 8.22882, 0.47043, 0.75828, -0.15972,
            0.95123, 3.2217, 5.90689, 0.1874, 0.69101, 0.05474,
            1.29049, 2.04403, 5.90689, 0.15282, 0.6874, -0.08952,
            1.43714, 4.67391, 8.22882, 0.3139, 0.68036, -0.232,
            0.5772, 3.83875, 9.81378, 0.59337, 0.74467, -0.39591,
            0.44473, 5.03957, 8.22882, 0.59681, 0.74844, -0.30756,
            0.96417, 3.94071, 8.22882, 0.29269, 0.75475, -0.13377,
            1.38843, 2.67497, 9.81378, 0.28718, 0.68217, -0.23533,
            1.09279, 2.47183, 8.22882, 0.66759, 0.75058, -0.32414,
            0.55303, 2.62904, 8.22882, 0.61708, 0.72563, -0.22741,
            0.58767, 3.06429, 5.90689, 0.2296, 0.70538, -0.03693,
            0.80884, 3.61914, 5.90689, 0.17468, 0.7288, 0.08665,
            3.18696, 4.74322, 5.90689, 0.21248, 0.60127, -0.30469,
            0.15607, 2.31525, 5.90689, 0.30546, 0.74026, -0.12827,
            2.1131, 4.07142, 5.90689, 0.23922, 0.65502, -0.2508,
            0.31468, 3.95962, 9.81378, 0.23977, 0.68024, 0.07997,
            1.07171, 0.93539, 5.90689, 0.17727, 0.71881, 0.06687,
            1.52743, 3.86696, 5.90689, 0.23479, 0.68857, -0.19049,
            0.99747, 2.03007, 8.22882, 0.14534, 0.65793, 0.02129,
            1.79361, 3.90471, 5.90689, 0.33936, 0.63822, -0.21177,
            0.90474, 3.9711, 5.90689, 0.31369, 0.75441, -0.12429,
            0.69201, 1.86352, 5.90689, 0.2308, 0.75579, -0.09061,
            1.53139, 4.5665, 9.81378, 0.33141, 0.60769, -0.1786,
            0.71681, 2.80183, 5.90689, 0.17477, 0.73258, 0.05078,
            1.48277, 0.97319, 5.90689, 0.6009, 0.76222, 0.26721,
            0.23797, 2.60064, 5.90689, 0.28621, 0.65685, 0.09197,
            2.11956, 3.50902, 8.22882, 0.22699, 0.72768, -0.02761,
            1.95651, 2.66475, 5.90689, 0.3454, 0.76274, 0.1063,
            1.76844, 0.46508, 5.90689, 0.25035, 0.64362, -0.21953,
            0.9505, 3.90647, 5.90689, 0.30739, 0.69937, -0.11649,
            0.51998, 1.60305, 5.90689, 0.443, 0.63319, -0.16691,
            1.45464, 2.77633, 5.90689, 0.18055, 0.75105, -0.07485,
            0.21369, 1.04795, 8.22882, 0.15779, 0.70683, -0.0281,
            0.31017, 1.46394, 5.90689, 0.19297, 0.66404, -0.10798,
            1.57646, 3.12124, 5.90689, 0.38833, 0.76082, 0.116,
            1.23422, 3.10944, 5.90689, 0.34671, 0.7603, -0.11627,
            0.97674, 3.08338, 5.90689, 0.1834, 0.63591, -0.03204,
            1.36291, 4.6631, 9.81378, 0.40252, 0.76502, -0.13821,
            1.39748, 1.70841, 5.90689, 0.14339, 0.63919, -0.05348,
            0.23677, 1.67316, 8.22882, 0.31892, 0.73426, 0.07608,
            0.44755, 4.01509, 8.22882, 0.12754, 0.47364, 0.00083,
            1.41144, 1.16484, 5.90689, 0.33777, 0.74488, -0.02056,
            0.93865, 3.06014, 5.90689, 0.25079, 0.71857, -0.08024,
            1.17291, 2.25683, 5.90689, 0.20829, 0.74591, -0.0825,
            2.4129, 3.47295, 5.90689, 0.23337, 0.75075, -0.04244,
            0.41994, 3.93864, 5.90689, 0.18573, 0.68103, -0.06561,
            0.56875, 1.64645, 8.22882, 0.35225, 0.67257, 0.1415,
            0.91896, 2.77617, 5.90689, 0.13789, 0.4793, -0.03639,
            2.12763, 4.4261, 5.90689, 0.22792, 0.74965, -0.03128,
            1.00982, 3.75779, 5.90689, 0.23866, 0.62414, -0.11483,
            0.20821, 3.58831, 5.90689, 0.30201, 0.75926, -0.16599,
            2.63481, 4.86259, 8.22882, 0.39299, 0.61222, -0.26677,
            1.06095, 1.43343, 5.90689, 0.1409, 0.65654, 0.0181,
            1.20073, 3.05619, 8.22882, 0.15847, 0.74223, -0.01107,
            0.28742, 0.10032, 5.90689, 0.38226, 0.73975, 0.16386,
            2.55088, 5.2846, 9.81378, 0.22262, 0.68554, -0.1457,
            1.70581, 4.14201, 5.90689, 0.78774, 0.74339, -0.39543,
            0.96982, 3.90101, 5.90689, 0.76043, 0.74823, -0.37184,
            2.76255, 5.37058, 9.81378, 0.39914, 0.75112, -0.05089,
            2.0265, 5.26741, 9.81378, 0.38933, 0.75962, -0.06909,
            0.861, 0.84706, 5.90689, 0.22618, 0.76139, 0.06608,
            0.58866, 1.97982, 8.22882, 0.34002, 0.72184, -0.17537,
            1.49968, 2.40768, 5.90689, 0.50536, 0.7595, -0.1719,
            0.49787, 2.04162, 5.90689, 0.31409, 0.72963, -0.05383,
            1.08113, 3.554, 5.90689, 0.43982, 0.76463, -0.22792,
            0.22234, 4.13179, 5.90689, 0.57189, 0.75376, -0.31586,
            0.72937, 0.89616, 5.90689, 0.16007, 0.67966, 0.04768,
            1.35456, 3.80684, 9.81378, 0.55764, 0.76804, 0.28193,
            1.49927, 4.76719, 8.22882, 0.268, 0.69717, 0.05914,
            0.71786, 3.50795, 9.81378, 0.40034, 0.72139, -0.25173,
            0.55082, 3.36779, 5.90689, 0.57662, 0.74668, 0.23703,
            2.22968, 3.86145, 5.90689, 0.27456, 0.65344, -0.10496,
            0.34839, 4.14096, 8.22882, 0.3123, 0.68218, 0.06872,
            0.6979, 1.51869, 5.90689, 0.38718, 0.74697, -0.1261,
            0.92298, 1.6536, 5.90689, 0.29428, 0.75403, 0.1014,
            0.27118, 3.96963, 5.90689, 0.24704, 0.71812, 0.13873,
            0.14986, 2.3135, 5.90689, 0.12957, 0.71036, 0.01206,
            1.08818, 1.02164, 5.90689, 0.19975, 0.62994, -0.13332,
            0.3325, 1.79667, 5.90689, 0.11678, 0.6885, -0.00895,
            2.60909, 4.09424, 5.90689, 0.16604, 0.60029, -0.13994,
            0.19588, 3.06212, 5.90689, 0.19148, 0.68441, -0.0934,
            0.26852, 1.51175, 8.22882, 0.16511, 0.59087, 0.00942,
            0.90885, 3.50144, 5.90689, 0.23765, 0.75272, -0.15811,
            1.70971, 4.87817, 9.81378, 0.16511, 0.65687, -0.04512,
            1.53354, 2.39836, 9.81378, 0.42433, 0.75215, -0.09585,
            0.27181, 1.82616, 5.90689, 0.15414, 0.62857, -0.0491,
            2.33912, 5.54613, 9.81378, 0.30486, 0.70543, 0.04638,
            0.226, 4.41501, 5.90689, 0.20108, 0.66969, 0.05241,
            0.20233, 1.6319, 5.90689, 0.25801, 0.74151, -0.1563,
            0.78261, 4.74496, 8.22882, 0.71455, 0.76168, -0.42145,
            0.41135, 4.63111, 8.22882, 0.39508, 0.63329, -0.1443,
            1.00839, 4.56805, 8.22882, 0.34527, 0.65712, -0.14636,
            1.29749, 4.35028, 8.22882, 0.4319, 0.65804, -0.1475,
            0.88186, 1.70725, 8.22882, 0.49813, 0.75238, 0.13464,
            0.94768, 5.32878, 9.81378, 0.26567, 0.69816, -0.04522,
            0.67767, 3.17978, 5.90689, 0.22461, 0.70222, -0.15137,
            1.56339, 2.78208, 5.90689, 0.21663, 0.67166, 0.16105,
            0.29317, 2.95241, 9.81378, 0.48256, 0.75464, -0.2734,
            1.5603, 3.60361, 9.81378, 0.25566, 0.74612, -0.07011,
            0.59072, 0.94618, 5.90689, 0.26436, 0.76329, -0.11859,
            1.55946, 3.15674, 5.90689, 0.17514, 0.68994, 0.0439,
            1.33402, 0.47426, 5.90689, 0.17326, 0.71569, -0.05291,
            0.29258, 4.11022, 9.81378, 0.3471, 0.75878, -0.16777,
            0.31054, 0.28088, 5.90689, 0.30083, 0.73984, -0.17104,
            2.21944, 3.31697, 5.90689, 0.17937, 0.75119, 0.00426,
            2.03511, 2.62681, 5.90689, 0.30802, 0.74538, -0.02538,
            2.5561, 3.9439, 5.90689, 0.28603, 0.75516, 0.031,
            0.83788, 3.7392, 5.90689, 0.25405, 0.74343, 0.16597,
            0.99634, 2.66759, 5.90689, 0.12913, 0.61673, -0.00161,
            0.17783, 3.29928, 5.90689, 0.13621, 0.56495, -0.01141,
            0.38921, 4.20122, 9.81378, 0.22576, 0.68868, -0.09854,
            2.06209, 1.82056, 5.90689, 0.21346, 0.74771, -0.03446,
            1.25643, 3.30462, 5.90689, 0.34176, 0.66025, -0.16318,
            0.41214, 3.38112, 9.81378, 0.17026, 0.71337, -0.03576,
            1.49712, 2.24749, 5.90689, 0.32495, 0.75217, 0.13182,
            0.14385, 1.53134, 5.90689, 0.1831, 0.71904, -0.06513,
            0.29977, 2.25179, 8.22882, 0.14414, 0.50515, -0.0025,
            1.0801, 3.75829, 9.81378, 0.33976, 0.76559, 0.0829,
            0.20642, 4.6258, 9.81378, 0.16962, 0.6175, -0.00584,
            1.77341, 3.05249, 5.90689, 0.36284, 0.7103, 0.06309,
            0.21473, 2.68246, 5.90689, 0.33353, 0.74198, -0.20756,
            2.17475, 2.90953, 5.90689, 0.25988, 0.75532, 0.05445,
            1.89416, 3.87532, 5.90689, 0.1768, 0.74793, -0.08789,
            0.49378, 1.97202, 9.81378, 0.21795, 0.63507, -0.07836,
            1.11155, 3.46826, 9.81378, 0.18954, 0.68293, 0.00795,
            1.3038, 3.78705, 5.90689, 0.37384, 0.75084, -0.02861,
            1.51274, 3.93996, 5.90689, 0.46666, 0.73765, -0.32424,
            0.18835, 2.47855, 5.90689, 0.12697, 0.65052, 0.00153,
            0.38195, 0.74751, 5.90689, 0.28728, 0.68173, 0.05477,
            1.93746, 2.40084, 5.90689, 0.53016, 0.76176, 0.19671,
            1.92622, 2.39835, 5.90689, 0.34177, 0.71655, -0.32606,
            0.34906, 4.42737, 9.81378, 0.40478, 0.7563, 0.21179,
            0.2757, 2.23809, 5.90689, 0.26945, 0.73458, 0.05438,
            1.55088, 4.76347, 8.22882, 0.32703, 0.74486, -0.13118,
            1.01549, 2.37571, 5.90689, 0.18138, 0.7522, 0.01305,
            0.27651, 3.09757, 5.90689, 0.33482, 0.72724, 0.2134,
            1.59756, 2.47482, 5.90689, 0.24995, 0.76305, 0.07862,
            0.2644, 4.29488, 8.22882, 0.26147, 0.75052, 0.11753,
            0.29016, 1.65445, 5.90689, 0.22943, 0.73767, -0.0101,
            0.55473, 0.72605, 5.90689, 0.49043, 0.73912, 0.34536,
            0.7141, 2.58846, 5.90689, 0.18244, 0.7024, -0.07195,
            2.01366, 3.97885, 8.22882, 0.21156, 0.73768, -0.17814,
            1.33764, 0.82299, 5.90689, 0.38419, 0.70972, 0.28832,
            1.15634, 1.66299, 5.90689, 0.24344, 0.73153, -0.19571,
            0.82573, 3.3491, 5.90689, 0.3067, 0.71494, 0.18455,
            0.19609, 3.75521, 5.90689, 0.22754, 0.70411, -0.13488,
            0.45173, 3.09427, 5.90689, 0.44215, 0.68288, -0.18607,
            1.10107, 2.97078, 5.90689, 0.26393, 0.70376, 0.05361,
            1.6498, 3.90825, 5.90689, 0.17367, 0.64489, 0.08464,
            0.27575, 3.98455, 8.22882, 0.51751, 0.74905, -0.28236,
            1.22644, 1.1627, 5.90689, 0.17947, 0.70964, -0.13651,
            2.22772, 4.90553, 8.22882, 0.28799, 0.66741, -0.17963,
            1.22951, 4.64272, 9.61566, 0.26539, 0.56544, 0.10279,
            1.02809, 3.71893, 9.81378, 0.22048, 0.61385, 0.13057,
            1.05045, 2.43767, 5.90689, 0.21182, 0.71726, -0.00335,
            0.89162, 2.87368, 8.22882, 0.27397, 0.71547, 0.14146,
            2.01329, 2.65054, 5.90689, 0.24548, 0.64776, -0.28231,
            0.91185, 2.28201, 5.90689, 0.45496, 0.69968, 0.18674,
            0.52577, 3.52408, 5.90689, 0.36072, 0.73114, -0.0267,
            1.6783, 2.74269, 5.90689, 0.63092, 0.76127, -0.24287,
            0.49726, 1.68515, 8.22882, 0.27579, 0.73666, 0.17712,
            3.14409, 4.87582, 5.90689, 0.30037, 0.6755, -0.41602,
            1.49641, 2.67855, 5.90689, 0.17876, 0.7176, -0.11447,
            1.79713, 2.55019, 5.90689, 0.21175, 0.69382, -0.05163,
            0.27326, 1.22173, 8.29626, 0.41234, 0.74946, -0.2497,
            2.29591, 5.22441, 9.81378, 0.43777, 0.69764, -0.44773,
            1.67383, 4.32182, 8.22882, 0.43921, 0.76056, 0.08555,
            1.98912, 3.6255, 5.90689, 0.49311, 0.72521, 0.15107,
            1.79708, 3.57705, 5.90689, 0.49605, 0.75954, -0.12868,
            1.07996, 3.2375, 8.22882, 0.33927, 0.73535, -0.0111,
            1.64341, 1.56343, 5.90689, 0.17652, 0.5335, -0.11065,
            1.55026, 1.54909, 8.22882, 0.20995, 0.62809, -0.1539,
            0.75869, 3.42188, 5.90689, 0.30666, 0.74607, -0.18852,
            0.25163, 0.55203, 8.22882, 0.15435, 0.73526, 0.03192,
            1.88148, 4.32333, 5.90689, 0.20897, 0.6805, -0.02648,
            0.53504, 0.87747, 5.90689, 0.14185, 0.6821, -0.03113,
            0.23123, 0.84669, 5.90689, 0.22411, 0.7106, -0.00232,
            1.33274, 1.21589, 8.22882, 0.49935, 0.76391, 0.24454,
            2.74728, 4.3415, 5.90689, 0.20987, 0.7049, -0.02715,
            1.01683, 4.11426, 8.22882, 0.15154, 0.60803, -0.00467,
            0.41036, 2.73972, 5.90689, 0.30874, 0.6478, -0.1196,
            1.38117, 0.81232, 5.90689, 0.27, 0.69015, 0.01507,
            1.93267, 3.30848, 5.90689, 0.4069, 0.6945, -0.22531,
            2.63375, 4.11607, 5.90689, 0.16264, 0.67275, -0.14281,
            1.16804, 2.82502, 5.90689, 0.17061, 0.65794, 0.07417,
            0.86824, 4.37489, 9.81378, 0.18087, 0.65041, -0.01637,
            1.29023, 2.36326, 5.90689, 0.20748, 0.67266, -0.16878,
            0.16539, 1.38815, 5.90689, 0.1553, 0.55771, -0.03005,
            0.45287, 2.3938, 9.81378, 0.16207, 0.56999, -0.00464,
            1.27772, 1.28863, 5.90689, 0.1274, 0.69613, -0.03061,
            2.63023, 3.99009, 5.90689, 0.42071, 0.7538, -0.12954,
            0.70445, 4.26013, 5.90689, 0.18177, 0.71375, 0.02871,
            1.67812, 1.49468, 5.90689, 0.17154, 0.70867, 0.09621,
            1.30273, 3.34268, 5.90689, 0.27728, 0.5871, 0.01493,
            1.73332, 4.06476, 9.81378, 0.30373, 0.65328, -0.30365,
            1.43089, 1.47419, 8.22882, 0.15979, 0.69745, 0.05786,
            1.69385, 2.46612, 9.36093, 0.48667, 0.71389, -0.39829,
            2.85244, 4.57049, 5.90689, 0.5268, 0.68822, -0.35845,
            2.56765, 4.65066, 5.90689, 0.63928, 0.72988, -0.39096,
            1.05145, 3.86589, 8.22882, 0.53163, 0.75821, 0.20275,
            1.34939, 3.01388, 5.90689, 0.26352, 0.75401, 0.06582,
            0.84155, 4.66599, 5.90689, 0.19623, 0.66788, -0.01995,
            1.10739, 0.86328, 5.90689, 0.18436, 0.66991, -0.10452,
            1.56848, 0.94901, 5.90689, 0.17418, 0.74797, -0.07106,
            1.4783, 2.04212, 9.81378, 0.25222, 0.70854, -0.13507,
            2.04943, 1.62322, 9.81378, 0.25991, 0.68751, -0.16216,
            0.19741, 3.52348, 8.22882, 0.2832, 0.728, 0.11309,
            1.48352, 3.69885, 8.22882, 0.36612, 0.75588, 0.13211,
            2.2124, 3.24207, 5.90689, 0.20315, 0.7441, -0.15476,
            0.44079, 0.30807, 8.22882, 0.26663, 0.73234, 0.14744,
            0.69215, 1.3861, 5.90689, 0.31277, 0.7501, 0.17781,
            0.83565, 1.91867, 8.22882, 0.26284, 0.74868, -0.11096,
            0.81625, 2.25708, 5.90689, 0.33429, 0.76113, 0.13682,
            1.21175, 3.75092, 5.90689, 0.52528, 0.75928, 0.27533,
            1.75318, 3.32002, 5.90689, 0.28015, 0.72392, -0.22448,
            0.15843, 0.50658, 5.90689, 0.22243, 0.73813, -0.10867,
            1.47786, 5.18665, 8.22882, 0.40454, 0.74795, -0.27946,
            1.61349, 3.23614, 8.22882, 0.61308, 0.7623, 0.25706,
            0.57866, 3.24362, 5.90689, 0.25742, 0.76325, -0.09785,
            1.24933, 4.18347, 8.22882, 0.18722, 0.72287, 0.04749,
            1.18863, 1.07701, 9.81378, 0.2047, 0.67261, -0.01799,
            0.68, 2.83097, 5.90689, 0.18733, 0.73629, -0.10678,
            1.4091, 4.35962, 5.90689, 0.15409, 0.67224, -0.07252,
            0.64022, 0.39711, 8.22882, 0.22487, 0.71234, -0.10096,
            1.56346, 4.39127, 8.22882, 0.17574, 0.63174, -0.10013,
            1.17469, 3.05248, 5.90689, 0.19657, 0.5213, -0.08667,
            0.29756, 1.02124, 5.90689, 0.23035, 0.58905, 0.04943,
            0.93357, 1.20419, 5.90689, 0.32095, 0.7227, -0.28222,
            1.07597, 1.418, 5.90689, 0.24514, 0.74995, -0.0983,
            1.61525, 4.8223, 9.81378, 0.20804, 0.74819, -0.06003,
            0.26392, 2.84949, 5.90689, 0.22188, 0.7215, -0.10162,
            1.41307, 0.64109, 5.90689, 0.44353, 0.75584, -0.22929,
            0.22001, 3.2867, 5.90689, 0.34762, 0.75031, -0.09765,
            0.5122, 2.40559, 8.22882, 0.26674, 0.72228, -0.02488,
            0.68189, 3.0932, 8.22882, 0.19387, 0.75065, 0.02826,
            0.48403, 1.72746, 5.90689, 0.15106, 0.73575, -0.01683,
            1.3075, 3.7091, 5.90689, 0.34787, 0.71896, -0.1901,
            0.73936, 2.69125, 5.90689, 0.22979, 0.69887, 0.14937,
            1.32142, 3.08334, 5.90689, 0.35251, 0.70784, -0.05868,
            1.80942, 4.20245, 5.90689, 0.31847, 0.7113, -0.10404,
            1.22498, 2.80533, 5.90689, 0.15281, 0.7186, 0.05998,
            0.18047, 2.21952, 5.90689, 0.24434, 0.7008, -0.12463,
            1.19285, 0.5829, 5.90689, 0.59321, 0.76329, -0.25903,
            1.02482, -0.0, 5.90689, 0.18099, 0.73232, 0.04054,
            0.76223, 2.62681, 5.90689, 0.17147, 0.58718, 0.05751,
            0.28036, 4.35818, 5.90689, 0.1578, 0.71045, -0.04472,
            0.31328, 3.49842, 5.90689, 0.18844, 0.63053, 0.05384,
            1.55181, 4.83716, 9.81378, 0.25145, 0.74481, 0.06721,
            1.35986, 0.69666, 8.22882, 0.16463, 0.73319, -0.01291,
            0.21968, 2.75132, 5.90689, 0.14328, 0.74855, 0.03235,
            0.48318, 0.30185, 5.90689, 0.30997, 0.69038, -0.08532,
            1.3209, 3.09296, 8.22882, 0.19243, 0.69139, 0.07121,
            1.82407, 1.83359, 5.90689, 0.40377, 0.76203, 0.15041,
            1.24528, 1.68962, 8.22882, 0.36079, 0.75588, 0.13876,
            1.99718, 3.80115, 5.90689, 0.26577, 0.75194, 0.04149,
            0.31582, 2.57062, 9.81378, 0.18322, 0.73336, 0.03323,
            0.84841, 3.4445, 5.90689, 0.37574, 0.74338, -0.28353,
            0.20398, 1.31604, 5.90689, 0.30682, 0.76003, -0.17496,
            1.22138, 1.84917, 9.81378, 0.37315, 0.7601, 0.16453,
            0.27446, 3.42491, 5.90689, 0.30722, 0.75782, 0.1708,
            2.975, 4.39167, 5.90689, 0.20235, 0.73237, -0.12791,
            0.21041, 3.91306, 8.22882, 0.35721, 0.7425, 0.19707,
            1.19277, 2.8046, 5.90689, 0.21586, 0.674, -0.04764,
            0.22439, 2.37658, 5.90689, 0.1857, 0.61447, 0.05661,
            0.93314, 0.57358, 8.22882, 0.19908, 0.52886, -0.06769,
            0.35257, 3.57572, 5.90689, 0.1873, 0.58111, -0.08956,
            1.37745, 2.19989, 5.90689, 0.40852, 0.76598, -0.16814,
            0.22531, 1.18066, 5.90689, 0.31081, 0.72122, -0.08619,
            0.30906, 3.63782, 5.90689, 0.3585, 0.69702, -0.17221,
            0.3022, 4.11885, 5.90689, 0.24877, 0.75158, -0.11001,
            1.4718, 3.453, 5.90689, 0.19558, 0.731, -0.14168,
            0.19196, 2.90088, 5.90689, 0.23029, 0.72416, 0.12242,
            0.75069, 1.89426, 5.90689, 0.21576, 0.68235, -0.01259,
            0.52209, 1.58089, 5.90689, 0.13281, 0.65432, 0.00973,
            1.74486, 2.7401, 8.22882, 0.4093, 0.7609, -0.09748,
            0.81349, 3.21275, 8.22882, 0.36063, 0.76214, -0.15432,
            0.30552, 0.0, 5.90689, 0.22513, 0.71072, 0.10313,
            0.23899, -0.0, 5.90689, 0.15369, 0.65326, 0.01302,
            0.48753, 3.45619, 5.90689, 0.20189, 0.67726, 0.10645,
            0.99203, 4.64603, 9.81378, 0.27556, 0.70668, 0.18124,
            0.26655, 2.85636, 5.90689, 0.45772, 0.74524, 0.18361,
            0.85412, 2.39121, 5.90689, 0.2331, 0.68522, -0.12227,
            1.2695, 4.24375, 5.90689, 0.1571, 0.57721, -0.01744,
            0.26953, 1.27344, 5.90689, 0.21899, 0.64335, 0.08405,
            0.3986, 2.28315, 5.90689, 0.60123, 0.7379, 0.2946,
            1.99885, 1.6759, 8.31687, 0.22657, 0.74616, -0.07667,
            1.05088, 1.76379, 8.22882, 0.2642, 0.70385, 0.01636,
            1.14191, 2.10034, 5.90689, 0.25212, 0.70963, 0.17573,
            1.56376, 4.20871, 8.22882, 0.28802, 0.75292, 0.04744,
            1.18609, 2.66474, 5.90689, 0.28166, 0.734, -0.11311,
            0.55123, 2.69105, 5.90689, 0.13255, 0.57007, -0.0264,
            0.80035, 3.21314, 8.22882, 0.2816, 0.71529, -0.14607,
            1.89697, 3.04077, 5.90689, 0.16231, 0.55777, -0.11622,
            0.24534, 3.12854, 5.90689, 0.30143, 0.73019, 0.11274,
            2.00038, 4.56504, 8.22882, 0.53343, 0.75907, -0.17329,
            1.63181, 3.84637, 5.90689, 0.30869, 0.72544, 0.26769,
            0.28456, 3.16692, 8.22882, 0.12683, 0.6727, -0.00818,
            1.77414, 1.79202, 5.90689, 0.413, 0.74665, 0.02926,
            2.6273, 4.53368, 5.90689, 0.20908, 0.69272, -0.22529,
            1.38221, 4.55246, 9.81378, 0.49086, 0.73809, 0.00958,
            0.85912, 0.0, 5.90689, 0.21015, 0.70867, -0.0742,
            2.59151, 4.45165, 5.90689, 0.35878, 0.66842, -0.23055,
            1.2369, 0.32006, 5.90689, 0.14682, 0.62594, -0.05684,
            0.47342, 4.99555, 9.81378, 0.36253, 0.73999, -0.26334,
            1.13361, 1.86467, 5.90689, 0.15213, 0.71495, 0.01676,
            2.07473, 5.11708, 8.22882, 0.23691, 0.73734, -0.07464,
            0.60347, 3.83818, 5.90689, 0.37583, 0.72936, -0.16735,
            0.26669, 4.02254, 5.90689, 0.43568, 0.75125, -0.17201,
            2.04571, 4.11753, 8.22882, 0.19623, 0.51096, 0.09668,
            2.0681, 3.86502, 5.90689, 0.35463, 0.67036, -0.39649,
            1.0194, 1.57268, 5.90689, 0.26943, 0.68249, -0.20182,
            0.88213, 0.47929, 5.90689, 0.24064, 0.55797, -0.10566,
            0.51004, 3.10661, 5.90689, 0.29389, 0.75309, 0.05701,
            0.27065, 4.25606, 8.22882, 0.19167, 0.67941, 0.06203,
            1.78758, 0.11552, 5.90689, 0.34765, 0.74349, -0.00809,
            0.88205, 4.29347, 8.22882, 0.27642, 0.70249, 0.18262,
            1.45889, 3.99377, 5.90689, 0.32089, 0.753, -0.16762,
            0.77206, 1.44194, 5.90689, 0.31792, 0.6404, -0.13254,
            0.21892, 2.2643, 5.90689, 0.18887, 0.71796, 0.03717,
            0.65933, 2.24991, 8.22882, 0.41932, 0.7519, -0.17849,
            0.40126, 2.19813, 8.22882, 0.15062, 0.63061, -0.0404,
            0.95437, 4.58602, 8.22882, 0.18506, 0.69926, -0.0632,
            3.36432, 4.67197, 5.90689, 0.18065, 0.50524, 0.18797,
            1.30776, 3.52065, 8.22882, 0.42245, 0.70225, 0.01183,
            0.28399, 4.13387, 9.81378, 0.35729, 0.6851, -0.13381,
            1.26159, 4.40248, 5.90689, 0.16508, 0.73302, 0.01309,
            0.95803, 1.95164, 8.29222, 0.57538, 0.77221, -0.25351,
            1.72309, 3.88603, 5.90689, 0.24307, 0.74357, -0.1935,
            1.25187, 4.28284, 8.22882, 0.29816, 0.60011, -0.152,
            1.26199, 1.24368, 5.90689, 0.17745, 0.60519, 0.08088,
            0.23817, 3.22319, 5.90689, 0.20213, 0.7553, -0.07343,
            0.81253, 3.40099, 8.22882, 0.1985, 0.71711, -0.11562,
            0.28881, 3.01393, 5.90689, 0.13164, 0.61958, -0.02551,
            0.23605, 2.63076, 5.90689, 0.26425, 0.74965, -0.05935,
            1.11523, 3.78835, 8.22882, 0.39102, 0.72032, -0.2157,
            0.74207, 2.20609, 5.90689, 0.52299, 0.7327, -0.15681,
            1.28102, 1.3614, 5.90689, 0.3483, 0.76334, 0.16014,
            0.50004, 5.005, 9.81378, 0.59213, 0.75671, -0.2956,
            0.47249, 1.05212, 8.22882, 0.38191, 0.7449, 0.19346,
            1.15447, 4.23833, 8.22882, 0.13858, 0.66772, -0.02185,
            0.37894, 3.84254, 5.90689, 0.18977, 0.74892, 0.05916,
            0.36153, 3.8165, 8.22882, 0.18516, 0.7387, -0.05858,
            0.37552, 2.99389, 8.22882, 0.24436, 0.74899, -0.11526,
            1.19357, 0.71839, 9.81378, 0.30643, 0.67527, -0.23279,
            1.49909, 1.06528, 5.90689, 0.40152, 0.76587, -0.12185,
            1.81357, 2.78269, 5.90689, 0.46042, 0.76036, 0.07811,
            0.84155, 5.02246, 8.22882, 0.3502, 0.74616, -0.18593,
            0.2561, 2.92015, 8.22882, 0.33656, 0.55916, -0.11538,
            0.72057, 3.94404, 8.22882, 0.33571, 0.56435, -0.12525,
            0.46604, 1.12033, 5.90689, 0.33172, 0.62823, 0.09474,
            0.53115, 3.52282, 5.90689, 0.27721, 0.57071, -0.09074,
            0.93083, 0.31461, 8.40493, 0.379, 0.61645, -0.18129,
            0.3192, -0.0, 5.90689, 0.21816, 0.74694, 0.09233,
            0.54549, 2.98828, 5.90689, 0.14609, 0.71075, -0.05473,
            0.85048, 1.42828, 5.90689, 0.50668, 0.74717, -0.23506,
            1.67444, 3.24517, 5.90689, 0.23661, 0.76619, 0.00962,
            2.16976, 4.11218, 5.90689, 0.55096, 0.75962, -0.22376,
            0.93718, 0.93191, 5.90689, 0.31129, 0.74099, 0.06998,
            0.56274, 1.36569, 5.90689, 0.17606, 0.75956, 0.05302,
            2.30326, 4.19949, 5.90689, 0.17799, 0.71408, 0.11676,
            0.33955, 5.90023, 9.81378, 0.25466, 0.57856, -0.16331,
            0.72327, 3.39362, 5.90689, 0.36737, 0.74246, 0.27929,
            1.67662, 4.07078, 5.90689, 0.23943, 0.61648, -0.15167,
            2.28554, 4.45196, 8.22882, 0.17709, 0.68636, 0.01854,
            1.67389, 4.248, 5.90689, 0.31594, 0.7598, 0.0845,
            1.85304, 2.96044, 5.90689, 0.21561, 0.73771, 0.05255,
            0.33118, 1.0202, 5.90689, 0.17548, 0.46792, 0.04138,
            0.40522, 3.79633, 8.22882, 0.32294, 0.74279, -0.01618,
            1.10475, 2.82728, 5.90689, 0.12437, 0.66692, -0.00101,
            0.84421, 0.42875, 5.90689, 0.18439, 0.62819, 0.06277,
            1.52565, 4.35588, 5.90689, 0.45994, 0.75136, -0.18116,
            2.04764, 5.40915, 8.22882, 0.55476, 0.73081, -0.44571,
            2.60451, 5.5609, 8.40493, 0.50532, 0.70332, -0.51144,
            0.24436, 0.98875, 5.90689, 0.22117, 0.69832, -0.10817,
            0.80232, 0.08664, 5.90689, 0.32512, 0.72279, 0.07321,
            0.1793, 2.7116, 5.90689, 0.26594, 0.75519, 0.12821,
            0.25708, 1.74546, 5.90689, 0.39294, 0.75766, -0.1741,
            1.00823, 2.86329, 5.90689, 0.21727, 0.74896, 0.08916,
            0.15462, 1.17472, 5.90689, 0.132, 0.70536, -0.00818,
            0.83767, 2.00688, 5.90689, 0.13652, 0.74764, 0.00204,
            0.33955, 4.19293, 8.22882, 0.37217, 0.69718, -0.12152,
            1.20491, 1.26033, 8.22882, 0.22457, 0.61997, 0.06158,
            0.20192, 0.72905, 5.90689, 0.15297, 0.62892, -0.03739,
            3.10937, 5.86198, 9.72055, 0.23082, 0.53598, -0.24027,
            1.57788, 4.89447, 8.22882, 0.22541, 0.70788, -0.10843,
            1.01529, 3.15635, 5.90689, 0.29824, 0.74371, -0.02149,
            2.5686, 4.25205, 5.90689, 0.27474, 0.65867, -0.35265,
            0.34542, 4.20649, 8.22882, 0.17869, 0.73574, 0.03516,
            2.03311, 4.56147, 8.22882, 0.3763, 0.71941, -0.31522,
            1.0378, 3.33733, 5.90689, 0.5317, 0.75639, -0.18757,
            0.76869, 2.24507, 5.90689, 0.22815, 0.58547, -0.08629,
            1.50772, 3.53271, 9.81378, 0.20214, 0.59393, -0.08649,
            0.95018, 3.99685, 9.81378, 0.18712, 0.5659, -0.08345,
            1.53364, 3.33425, 8.22882, 0.20713, 0.74352, -0.04049,
            0.44817, 2.85921, 5.90689, 0.1735, 0.67919, 0.00039,
            0.29921, 3.08277, 5.90689, 0.54028, 0.75371, -0.32674,
            0.56199, 1.18671, 5.90689, 0.15911, 0.71143, -0.05437,
            1.0008, 3.37714, 5.90689, 0.17753, 0.73713, -0.07492,
            0.60171, 4.13919, 5.90689, 0.26747, 0.754, 0.06555,
            0.41645, 3.80993, 5.90689, 0.14163, 0.63971, -0.03981,
            1.85362, 4.10935, 5.90689, 0.19591, 0.71719, -0.14274,
            1.73185, 2.25043, 8.22882, 0.25315, 0.66135, 0.20376,
            1.9076, 3.31209, 8.2599, 0.21811, 0.64094, -0.1574,
            1.47822, 2.11589, 8.22882, 0.19455, 0.74843, -0.01024,
            1.95793, 4.95736, 8.22882, 0.2663, 0.75906, 0.04896,
            2.72555, 4.40155, 5.90689, 0.26004, 0.73043, -0.27289,
            1.77176, 0.92265, 5.90689, 0.18487, 0.69864, -0.13454,
            0.70731, 2.07286, 8.22882, 0.23102, 0.68953, -0.10317,
            1.26645, 1.92151, 5.90689, 0.31832, 0.5732, -0.18636,
            0.20592, 0.89136, 5.90689, 0.18928, 0.74267, 0.07925,
            3.0399, 5.21932, 8.22882, 0.2741, 0.73619, -0.1483,
            1.22165, 1.7497, 8.22882, 0.36445, 0.71234, -0.25606,
            0.29549, 4.17078, 5.90689, 0.33967, 0.74569, 0.1344,
            1.7309, 3.95808, 5.90689, 0.26061, 0.70016, 0.0808,
            0.75046, 4.04012, 5.90689, 0.28338, 0.69648, 0.05845,
            1.10802, 2.32126, 5.90689, 0.2723, 0.76179, 0.10787,
            0.5037, 0.81265, 5.90689, 0.51802, 0.75262, 0.22553,
            0.31577, 3.01482, 5.90689, 0.49924, 0.72368, 0.26388,
            0.58418, 3.42785, 5.90689, 0.18076, 0.72344, -0.00853,
            0.57534, 2.63157, 9.81378, 0.28989, 0.70409, -0.15806,
            0.27164, 2.50694, 8.22882, 0.217, 0.73982, 0.10172,
            1.70533, 3.2829, 5.90689, 0.33727, 0.76015, -0.11038,
            0.32048, 5.11486, 9.81378, 0.19799, 0.73281, -0.00609,
            0.19164, 2.98043, 5.90689, 0.19678, 0.64807, -0.05979,
            0.83975, 2.74603, 5.90689, 0.4763, 0.75323, -0.25402,
            0.74568, -0.0, 5.90689, 0.22042, 0.74891, -0.08482,
            1.90162, 1.83382, 5.90689, 0.17109, 0.72701, -0.11274,
            0.30407, 1.44462, 5.90689, 0.35237, 0.74308, 0.13813,
            1.14075, 4.26404, 5.90689, 0.19948, 0.71263, -0.08565,
            0.21118, 0.0226, 5.90689, 0.12962, 0.73482, 0.01814,
            0.31622, 2.76173, 5.90689, 0.45788, 0.747, -0.22868,
            0.58522, 0.96599, 5.90689, 0.22746, 0.74312, 0.12367,
            0.50211, 1.94864, 5.90689, 0.27053, 0.71812, -0.18798,
            2.00734, 4.07431, 8.22882, 0.30945, 0.61643, -0.24527,
            1.56501, -0.0, 5.90689, 0.20593, 0.75418, -0.01948,
            1.28815, 0.52761, 5.90689, 0.37462, 0.6915, -0.23282,
            1.15484, 4.65729, 9.81378, 0.37976, 0.7089, -0.3312,
            1.42709, 2.1856, 8.3609, 0.14727, 0.51223, -0.03549,
            0.22452, 4.38144, 9.81378, 0.28894, 0.74624, 0.13592,
            0.17446, 3.28885, 5.90689, 0.37926, 0.75391, 0.22155,
            1.62957, 1.31484, 5.90689, 0.52555, 0.70014, -0.35232,
            1.10659, 1.78564, 5.90689, 0.27792, 0.7277, 0.00437,
            0.34005, 2.53583, 5.90689, 0.17021, 0.5046, -0.05478,
            1.42139, 2.88036, 5.90689, 0.13559, 0.67185, -0.06779,
            0.23465, 4.41445, 8.22882, 0.17673, 0.70557, -0.04719,
            1.23117, 2.85098, 5.90689, 0.4801, 0.71349, -0.33317,
            1.28689, 2.77776, 5.90689, 0.18672, 0.6176, 0.09796,
            1.90949, 3.0643, 5.90689, 0.14292, 0.65788, -0.05451,
            0.21015, 3.64792, 5.90689, 0.25797, 0.71743, 0.02921,
            1.7687, 2.52652, 5.90689, 0.28989, 0.67353, -0.14622,
            2.28478, 3.88079, 5.90689, 0.29514, 0.73232, -0.2142,
            0.7571, 3.25333, 5.90689, 0.20502, 0.65666, -0.10278,
            1.22009, 0.8371, 9.81378, 0.53643, 0.7685, 0.21423,
            0.97672, 2.14396, 9.81378, 0.60203, 0.7476, 0.23608,
            1.56709, 2.19858, 9.81378, 0.57287, 0.77072, 0.20258,
            0.23209, 3.62189, 5.90689, 0.31454, 0.66862, -0.09683,
            0.75798, 4.7068, 9.81378, 0.34014, 0.75617, -0.14985,
            0.50901, 2.78948, 9.81378, 0.38882, 0.74198, 0.23381,
            1.42829, 2.98173, 8.22882, 0.20104, 0.73205, -0.14675,
            1.12911, 2.35349, 5.90689, 0.24151, 0.76341, 0.00662,
            0.45077, 2.60784, 8.22882, 0.15823, 0.69037, 0.03165,
            1.47277, 0.02521, 5.90689, 0.23744, 0.73255, 0.06459,
            0.23635, 1.90172, 5.90689, 0.286, 0.74659, 0.16945,
            1.6402, 2.98362, 5.90689, 0.15412, 0.61513, 0.00069,
            1.3818, 3.1453, 5.90689, 0.20454, 0.5852, -0.11173,
            0.34867, 4.26611, 8.22882, 0.48763, 0.75719, -0.17803,
            0.94907, 4.8232, 8.22882, 0.23166, 0.73787, -0.12901,
            0.71158, 0.40164, 5.90689, 0.32151, 0.75776, 0.14099,
            1.11016, 2.06881, 5.90689, 0.58296, 0.75763, -0.32268
        };
        public static readonly double[] LongFrac = {
            0.55, 0.4719, 0.6316, 0.1984, 0.7436, 0.5, 0.0889, 0.3208, 0.3486, 0.6941, 0.5778, 0.8056,
            0.5714, 0.9118, 0.1538, 0.6514, 0.2692, 0.3784, 0.641, 0.765, 0.3975, 0.375, 0.5362, 0.1505,
            0.21, 0.3736, 0.6026, 0.7808, 0.1988, 0.2947, 0.5781, 0.3333, 0.6286, 0.5616, 0.5217, 0.5391,
            0.4789, 0.4923, 0.6471, 0.3871, 0.0714, 0.4929, 0.3443, 0.2062, 0.2759, 0.5517, 0.6832, 0.5854,
            0.6238, 0.8, 0.2714, 0.0694, 0.3929, 0.7294, 0.4737, 0.1923, 0.3036, 0.736, 0.1724, 0.4237,
            0.4811, 0.4783, 0.3178, 0.445, 0.4658, 0.7826, 0.1549, 0.191, 0.4043, 0.3165, 0.4872, 0.0,
            0.6667, 0.4133, 0.5926, 0.4737, 0.3465, 0.6168, 0.7722, 0.35, 0.4462, 0.4717, 0.3611, 0.319,
            0.3378, 0.1, 0.4945, 0.3099, 0.48, 0.3365, 0.1486, 0.2292, 0.2207, 0.7079, 0.2075, 0.4194,
            0.6462, 0.2, 0.3146, 0.5524, 0.2778, 0.5385, 0.1129, 0.3714, 0.5205, 0.7186, 0.885, 0.5179,
            0.4615, 0.6331, 0.5214, 0.4085, 0.359, 0.2959, 0.2025, 0.7705, 0.5161, 0.0833, 0.0769, 0.5155,
            0.7073, 0.7409, 0.1538, 0.4524, 0.4713, 0.4959, 0.775, 0.5, 0.1429, 0.4487, 0.7895, 0.134,
            0.2549, 0.5, 0.194, 0.2991, 0.1333, 0.5741, 0.2667, 0.4651, 0.2553, 0.2468, 0.2778, 0.1458,
            0.3077, 0.338, 0.4645, 0.5294, 0.602, 0.2284, 0.7941, 0.5897, 0.8, 0.4877, 0.32, 0.5585,
            0.525, 0.4949, 0.748, 0.5816, 0.4944, 0.2674, 0.2552, 0.5, 0.5278, 0.4592, 0.3721, 0.2951,
            0.5882, 0.4008, 0.7677, 0.5135, 0.1773, 0.4237, 0.5, 0.3357, 0.1644, 0.5068, 0.3712, 0.5281,
            0.4205, 0.6497, 0.6554, 0.7203, 0.3462, 0.4878, 0.5082, 0.8095, 0.6106, 0.1214, 0.4717, 0.4375,
            0.8392, 0.5429, 0.7273, 0.4097, 0.2041, 0.5135, 0.6267, 0.375, 0.3704, 0.6667, 0.54, 0.5169,
            0.6698, 0.6897, 0.4043, 0.4907, 0.2593, 0.5385, 0.5093, 0.2745, 0.6594, 0.4679, 0.1081, 0.0357,
            0.8182, 0.85, 0.55, 0.2713, 0.2121, 0.7619, 0.6134, 0.2222, 0.3924, 0.5135, 0.5042, 0.3672,
            0.4565, 0.432, 0.4674, 0.0, 0.5882, 0.5789, 0.1967, 0.5641, 0.5896, 0.3107, 0.1364, 0.7,
            0.2051, 0.4286, 0.7419, 0.1429, 0.625, 0.8182, 0.4167, 0.2768, 0.3399, 0.8085, 0.5229, 0.1688,
            0.4316, 0.5429, 0.4427, 0.3452, 0.6262, 0.2, 0.1979, 0.5294, 0.4714, 0.6667, 0.8947, 0.2925,
            0.2703, 0.0928, 0.5, 0.6667, 0.6415, 0.45, 0.2745, 0.5278, 0.5882, 0.6346, 0.9143, 0.1698,
            0.1429, 0.6842, 0.2424, 0.8841, 0.3962, 0.6279, 0.1484, 0.3158, 0.7879, 0.6788, 0.3478, 0.3415,
            0.2188, 0.2941, 0.2642, 0.2, 0.541, 0.2353, 0.75, 0.5957, 0.7101, 0.4097, 0.5596, 0.3462,
            0.223, 0.4675, 0.4862, 0.5556, 0.7059, 0.1448, 0.7971, 0.5308, 0.5, 0.1416, 0.4336, 0.36,
            0.7822, 0.3939, 0.5099, 0.7049, 0.8478, 0.3776, 0.5, 0.4522, 0.1667, 0.549, 0.2868, 0.7619,
            0.561, 0.4388, 0.5556, 0.4497, 0.75, 0.1471, 0.4361, 0.4085, 0.4615, 0.1852, 0.4, 0.3652,
            0.9375, 0.9412, 0.34, 0.411, 0.5, 0.6061, 0.4643, 0.2921, 0.1707, 0.2663, 0.4167, 0.4138,
            0.4025, 0.4694, 0.5, 0.2184, 0.4156, 0.5561, 0.5, 0.8143, 0.7377, 0.5787, 0.8252, 0.0909,
            0.5063, 0.4762, 0.4615, 0.4126, 0.4884, 0.4881, 0.1471, 0.5274, 0.4043, 0.5895, 0.5622, 0.6181,
            0.6286, 0.439, 0.5714, 0.6806, 0.3265, 0.4206, 0.6667, 0.6094, 0.5149, 0.1373, 0.4231, 0.1667,
            0.8088, 0.7225, 0.6916, 0.1333, 0.7308, 0.4167, 0.1639, 0.0583, 0.2593, 0.5882, 0.3261, 0.3622,
            0.0556, 0.5094, 0.4236, 0.5962, 0.0549, 0.4521, 0.4429, 0.3368, 0.56, 0.3925, 0.6667, 0.5088,
            0.4746, 0.6829, 0.4375, 0.5, 0.8083, 0.4198, 0.4138, 0.5422, 0.4722, 0.7889, 0.1892, 0.6,
            0.2849, 0.4654, 0.5321, 0.1982, 0.1111, 0.24, 0.3895, 0.1163, 0.5745, 0.2636, 0.614, 0.8915,
            0.0612, 0.4533, 0.5714, 0.3023, 0.1125, 0.3939, 0.4706, 0.5278, 0.4167, 0.2632, 0.7368, 0.7034,
            0.7515, 0.5974, 0.1273, 0.4175, 0.6548, 0.2444, 0.6954, 0.4527, 0.563, 0.7664, 0.2157, 0.2755,
            0.4426, 0.6379, 0.5481, 0.5217, 0.6875, 0.5066, 0.7135, 0.5556, 0.678, 0.5909, 0.3968, 0.42,
            0.1667, 0.7222, 0.5132, 0.5625, 0.1681, 0.1549, 0.5789, 0.1389, 0.1346, 0.7176, 0.95, 0.5455,
            0.5714, 0.2143, 0.1765, 0.2857, 0.4786, 0.5298, 0.1579, 0.0741, 0.5647, 0.4828, 0.5344, 0.0794,
            0.2483, 0.4453, 0.1745, 0.2402, 0.5234, 0.5652, 0.6092, 0.3598, 0.5684, 0.3465, 0.4706, 0.4348,
            0.3918, 0.8421, 0.3265, 0.7838, 0.7846, 0.7351, 0.1915, 0.5926, 0.6466, 0.1857, 0.5541, 0.3169,
            0.4699, 0.3657, 0.6714, 0.7537, 0.8529, 0.0526, 0.7576, 0.5, 0.7348, 0.7721, 0.3448, 0.3,
            0.5215, 0.2474, 0.2619, 0.5, 0.5902, 0.1368, 0.5298, 0.4839, 0.2871, 0.2537, 0.6923, 0.55,
            0.6154, 0.4561, 0.4595, 0.5745, 0.5306, 0.3049, 0.4312, 0.8462, 0.5521, 0.4634, 0.4884, 0.5208,
            0.7091, 0.2353, 0.2712, 0.56, 0.1333, 0.5446, 0.5385, 0.4211, 0.7619, 0.574, 0.7721, 0.0606,
            0.3469, 0.1067, 0.4375, 0.6619, 0.1233, 0.5789, 0.1695, 0.2388, 0.3636, 0.5, 0.6545, 0.5192,
            0.6265, 0.2542, 0.5385, 0.05, 0.3486, 0.4516, 0.2636, 0.5672, 0.3455, 0.4508, 0.5679, 0.3788,
            0.6111, 0.2832, 0.6, 0.1579, 0.5922, 0.4487, 0.3645, 0.2769, 0.44, 0.2973, 0.5455, 0.265,
            0.2989, 0.1762, 0.2667, 0.6905, 0.566, 0.9067, 0.2273, 0.0, 0.0667, 0.5556, 0.5909, 0.6164,
            0.5571, 0.4831, 0.5068, 0.3148, 0.25, 0.6992, 0.5366, 0.5862, 0.4762, 0.7778, 0.2375, 0.5,
            0.3162, 0.7176, 0.7664, 0.5977, 0.3291, 0.5667, 0.0577, 0.3713, 0.42, 0.2079, 0.2667, 0.5833,
            0.4604, 0.4412, 0.5077, 0.2897, 0.2632, 0.5882, 0.3913, 0.5, 0.5667, 0.6522, 0.3259, 0.8923,
            0.4727, 0.5526, 0.2237, 0.6614, 0.3617, 0.4407, 0.25, 0.4275, 0.2937, 0.3306, 0.9, 0.5197,
            0.5195, 0.4792, 0.3548, 0.2329, 0.6377, 0.5929, 0.4813, 0.5517, 0.5, 0.6346, 0.4521, 0.2727,
            0.416, 0.1603, 0.375, 0.4828, 0.6, 0.1125, 0.5363, 0.6441, 0.4167, 0.2157, 0.6833, 0.6154,
            0.5699, 0.5354, 0.73, 0.6513, 0.7184, 0.5695, 0.8649, 0.4317, 0.3231, 0.8654, 0.2667, 0.8317,
            0.2833, 0.5068, 0.5714, 0.8182, 0.3235, 0.2188, 0.1786, 0.4444, 0.5, 0.5238, 0.5312, 0.2075,
            0.7895, 0.5208, 0.6667, 0.5962, 0.0, 0.1852, 0.3077, 0.4, 0.2857, 0.3077, 0.4694, 0.7042,
            0.5273, 0.2059, 0.3226, 0.1386, 0.6154, 0.2632, 0.5102, 0.5872, 0.7778, 0.3378, 0.5233, 0.3878,
            0.5238, 0.14, 0.0769, 0.8468, 0.4795, 0.1333, 0.5625, 0.5938, 0.3394, 0.3636, 0.6211, 0.8788,
            0.619, 0.5357, 0.6744, 0.4762, 0.0526, 0.2222, 0.5507, 0.5586, 0.3871, 0.2656, 0.2526, 0.8889,
            0.9333, 0.6049, 0.5526, 0.15, 0.5952, 0.785, 0.6087, 0.6732, 0.7143, 0.1491, 0.2736, 0.1569,
            0.5385, 0.3371, 0.4574, 0.6304, 0.2906, 0.2366, 0.5294, 0.1294, 0.1429, 0.5741, 0.16, 0.3404,
            0.3968, 0.3991, 0.2692, 0.3878, 0.5802, 0.5254, 0.4873, 0.23, 0.7895, 0.5676, 0.3158, 0.7162,
            0.3448, 0.4186, 0.6125, 0.7353, 0.5091, 0.5093, 0.4167, 0.4412, 0.6813, 0.4737, 0.6825, 0.6543,
            0.716, 0.4351, 0.6415, 0.2632, 0.1935, 0.5758, 0.9, 0.16, 0.7333, 0.4851, 0.5978, 0.2857,
            0.4935, 0.4419, 0.4961, 0.2593, 0.2895, 0.1707, 0.7443, 0.5289, 0.5821, 0.6364, 0.5161, 0.7162,
            0.5876, 0.6719, 0.4722, 0.7368, 0.3361, 0.3521, 0.72, 0.7736, 0.4118, 0.5224, 0.871, 0.3415,
            0.407, 0.5347, 0.5532, 0.1233, 0.7622, 0.7551, 0.9556, 0.4767, 0.4304, 0.0769, 0.7273, 0.3387,
            0.0909, 0.3649, 0.4634, 0.5345, 0.371, 0.2821, 0.2214, 0.5333, 0.075, 0.1628, 0.3824, 0.7172,
            0.44, 0.3571, 0.7018, 0.2553, 0.473, 0.7227, 0.3696, 0.5522, 0.3725, 0.8333, 0.4815, 0.4286,
            0.5435, 0.25, 0.0833, 0.2571, 0.8732, 0.3016, 0.5122, 0.5, 0.3427, 0.5, 0.3962, 0.7939,
            0.2826, 0.4821, 0.3418, 0.7432, 0.5158, 0.4592, 0.4286, 0.6455, 0.2533, 0.4026, 0.5, 0.3846,
            0.5833, 0.38, 0.5, 0.8247, 0.4354, 0.2985, 0.4172, 0.4259, 0.6176, 0.7174, 0.8776, 0.6667,
            0.875, 0.1807, 0.4444, 0.4737, 0.5082, 0.625, 0.4421, 0.5491, 0.679, 0.6119, 0.125, 0.1,
            0.4054, 0.5417, 0.8366, 0.2623, 0.6503, 0.5787, 0.5307, 0.5789, 0.5641, 0.5301, 0.1875, 0.2826,
            0.4936, 0.0556, 0.4717, 0.186, 0.3936, 0.3243, 0.5, 0.6875, 0.4211, 0.6269, 0.1857, 0.3394,
            0.3262, 0.7273, 0.4155, 0.1484, 0.4348, 0.2857, 0.4483, 0.2727, 0.069, 0.0938, 0.5444, 0.303,
            0.8333, 0.2308, 0.4419, 0.7368, 0.6842, 0.5522, 0.6741, 0.75, 0.8889, 0.5025, 0.6061, 0.598,
            0.5385, 0.5763, 0.4557, 0.2391, 0.3875, 0.2048, 0.7794, 0.3206, 0.5357, 0.1933, 0.7114, 0.3154,
            0.3226, 0.2784, 0.3611, 0.4, 0.4583, 0.6061, 0.8678, 0.125, 0.5126, 0.4643, 0.2621, 0.4865,
            0.119, 0.7838, 0.1584, 0.6202, 0.2, 0.1354, 0.288, 0.8182, 0.75, 0.2353, 0.4267, 0.45,
            0.5098, 0.3765, 0.6012, 0.5172, 0.6667, 0.8182, 0.4881, 0.2371, 0.2222, 0.4286, 0.8252, 0.2041
        };
        public static readonly int[] MemberCount = {
            127, 184, 58, 126, 118, 88, 45, 160, 109, 85, 92, 73, 138, 35, 92, 112, 162, 117, 39, 188,
            163, 98, 142, 93, 105, 91, 82, 78, 170, 211, 66, 54, 73, 74, 115, 116, 193, 72, 175, 33,
            42, 140, 63, 97, 61, 213, 165, 41, 110, 122, 70, 74, 178, 87, 78, 158, 56, 128, 29, 60,
            108, 48, 107, 198, 147, 141, 72, 95, 47, 82, 197, 40, 18, 76, 110, 58, 105, 179, 79, 86,
            70, 159, 74, 123, 150, 80, 91, 76, 51, 107, 75, 49, 149, 100, 217, 31, 65, 115, 225, 146,
            108, 55, 62, 105, 73, 174, 113, 60, 119, 178, 123, 223, 207, 198, 80, 61, 65, 12, 13, 162,
            44, 199, 105, 42, 89, 121, 81, 86, 42, 78, 154, 98, 53, 125, 68, 111, 17, 55, 45, 45,
            147, 78, 56, 49, 91, 74, 163, 126, 105, 163, 106, 79, 137, 162, 50, 191, 42, 103, 128, 100,
            90, 86, 193, 55, 73, 101, 89, 124, 104, 252, 203, 115, 142, 61, 49, 147, 75, 74, 135, 91,
            89, 164, 153, 126, 165, 41, 64, 22, 118, 142, 54, 64, 148, 71, 126, 145, 147, 37, 79, 32,
            55, 76, 51, 240, 107, 29, 48, 111, 135, 54, 109, 204, 151, 115, 74, 86, 78, 20, 87, 130,
            100, 64, 123, 27, 83, 39, 122, 129, 147, 207, 186, 13, 69, 95, 62, 40, 136, 103, 66, 95,
            39, 50, 161, 23, 105, 35, 196, 116, 162, 148, 109, 78, 98, 183, 133, 85, 113, 16, 96, 52,
            70, 9, 20, 110, 111, 99, 137, 42, 161, 60, 51, 36, 190, 52, 40, 53, 42, 61, 33, 73,
            55, 44, 131, 38, 35, 206, 140, 41, 131, 17, 58, 16, 62, 36, 35, 48, 69, 151, 112, 55,
            148, 173, 114, 37, 110, 145, 72, 212, 54, 113, 113, 110, 105, 37, 208, 63, 48, 156, 85, 117,
            67, 55, 143, 221, 42, 101, 82, 193, 136, 34, 134, 147, 87, 81, 137, 182, 16, 17, 50, 74,
            171, 36, 115, 93, 41, 171, 48, 58, 170, 49, 41, 90, 80, 191, 131, 71, 63, 184, 105, 44,
            83, 21, 14, 151, 94, 85, 104, 163, 49, 97, 187, 147, 74, 168, 66, 81, 98, 107, 124, 66,
            101, 107, 26, 12, 76, 212, 107, 121, 26, 170, 123, 122, 162, 39, 46, 128, 60, 106, 149, 55,
            92, 79, 73, 97, 79, 196, 48, 233, 61, 41, 186, 84, 121, 82, 95, 85, 76, 96, 76, 80,
            178, 165, 157, 115, 45, 101, 95, 43, 50, 110, 64, 133, 98, 77, 50, 44, 80, 37, 69, 114,
            64, 57, 83, 154, 170, 160, 112, 107, 84, 45, 155, 156, 123, 110, 53, 200, 62, 180, 107, 72,
            71, 164, 187, 48, 210, 134, 63, 50, 121, 36, 76, 116, 122, 71, 179, 37, 52, 89, 20, 80,
            31, 14, 34, 21, 119, 155, 98, 54, 176, 91, 132, 64, 146, 133, 153, 187, 111, 70, 90, 223,
            96, 101, 51, 144, 100, 116, 99, 37, 67, 159, 49, 54, 119, 144, 77, 190, 84, 136, 146, 212,
            68, 38, 33, 132, 183, 142, 175, 20, 186, 100, 128, 34, 131, 95, 153, 65, 101, 67, 28, 62,
            28, 58, 76, 49, 49, 166, 109, 13, 100, 124, 43, 49, 167, 163, 59, 25, 30, 102, 27, 19,
            21, 172, 136, 33, 150, 75, 82, 140, 146, 57, 59, 135, 203, 24, 227, 54, 85, 60, 121, 41,
            111, 31, 132, 72, 116, 124, 163, 136, 54, 116, 60, 20, 103, 158, 207, 131, 156, 37, 33, 117,
            87, 210, 15, 128, 107, 76, 45, 11, 15, 10, 22, 155, 70, 91, 150, 112, 56, 140, 43, 58,
            43, 72, 80, 54, 120, 136, 109, 180, 80, 158, 104, 173, 52, 182, 46, 36, 149, 34, 65, 148,
            19, 17, 24, 15, 31, 46, 139, 66, 55, 76, 156, 128, 95, 64, 153, 135, 127, 126, 126, 137,
            79, 98, 96, 73, 72, 142, 166, 29, 70, 55, 75, 182, 128, 156, 40, 61, 101, 81, 188, 64,
            73, 53, 62, 192, 93, 235, 105, 153, 105, 154, 37, 197, 68, 53, 111, 103, 124, 73, 123, 77,
            68, 97, 56, 10, 22, 194, 68, 55, 57, 96, 46, 55, 28, 164, 107, 47, 28, 52, 50, 71,
            57, 34, 34, 169, 67, 95, 151, 115, 37, 77, 86, 100, 64, 50, 106, 112, 78, 124, 83, 32,
            116, 46, 95, 73, 42, 29, 44, 21, 19, 9, 69, 147, 62, 130, 98, 27, 15, 82, 79, 121,
            45, 113, 93, 155, 45, 116, 113, 51, 26, 178, 98, 47, 205, 94, 55, 88, 35, 57, 83, 147,
            66, 229, 53, 150, 84, 120, 168, 100, 118, 78, 76, 154, 178, 44, 86, 103, 112, 108, 60, 71,
            185, 62, 66, 86, 81, 133, 59, 137, 128, 39, 131, 101, 81, 135, 98, 14, 78, 135, 134, 109,
            192, 165, 184, 131, 141, 55, 94, 75, 103, 132, 39, 79, 125, 71, 81, 54, 34, 67, 96, 85,
            176, 104, 96, 73, 170, 49, 47, 92, 80, 66, 22, 67, 45, 74, 42, 179, 62, 117, 131, 15,
            40, 87, 34, 145, 76, 42, 57, 95, 74, 249, 46, 69, 102, 6, 29, 45, 92, 24, 109, 36,
            74, 189, 124, 168, 145, 57, 53, 137, 48, 57, 81, 148, 95, 100, 14, 116, 79, 78, 12, 15,
            48, 50, 18, 103, 213, 67, 164, 54, 106, 144, 50, 9, 84, 83, 50, 95, 127, 21, 98, 176,
            83, 67, 16, 10, 160, 80, 154, 126, 186, 204, 188, 59, 39, 86, 17, 92, 158, 38, 111, 43,
            94, 74, 18, 17, 95, 195, 70, 175, 233, 112, 144, 130, 23, 50, 88, 56, 58, 66, 90, 33,
            226, 39, 44, 95, 76, 69, 138, 74, 68, 202, 66, 106, 131, 59, 161, 93, 80, 85, 137, 132,
            92, 120, 165, 137, 31, 99, 38, 31, 24, 71, 131, 26, 127, 56, 148, 112, 44, 74, 104, 130,
            65, 96, 127, 11, 16, 17, 75, 62, 52, 86, 163, 91, 55, 172, 85, 100, 54, 79, 107, 52
        };
    }
// ===SHARED-CORE-V02 END===

    }
}

// -----------------------------------------------------------------------------
// CHANGELOG
//   v0.2.0-RC (2026-07-18, research/nt8_port P2b): DECISION CORE PORTED IN.
//     Resolves the v0.1 stub TODOs:
//       P2-1  22 generator bodies (Gens.cs) -> shared core (batch, driven by a
//             per-minute streaming harness over the day-so-far 5s buffer).
//       P2-2  frozen combiner embedded as constants (ModelData: 27 cols, coef, mu,
//             sd, top-K, threshold 0.7139834155227371) -- read from _model.json,
//             not retyped; byte-proven by the shim.
//       P2-3  native z_se as 1m endpoint-OLS (window 15, ddof 2). STILL FLAGGED for
//             bit-parity vs core_v2 _ols_fit_kernel before live (harness exported it).
//       P2-4  TMPL0 frozen codebook embedded (Tmpl0Data: 1020 templates) + nearest-
//             centroid + P2 same-bar tie rule (Core.ResolveTmpl0).
//       P2-6  zz_thr ATR(14 1m)x4 index//12 buckets -- inside Ctx.BuildZzThr (verbatim).
//       P2-7  ±180s same-direction consensus -- Core.ProcessDay (verbatim). Entries
//             act on the SETTLED minute (curMin-180s) so P is the full-window value.
//       P2-9  R-trigger reversal exit wired: confirmed pivot against the open leg
//             (per-minute zz_confirm, causal) flattens it.
//       P2-13 whole shared region + shim compile at LangVersion=7.3 (down-level proof).
//     Deviations vs harness forced by C#7.3/.NET4.8 (all proven parity-neutral by the
//     byte-identical shim output): named ValueTuples -> structs; Math.Log2 -> Pd.Log2;
//     JSON codebook load -> embedded constants; LINQ OrderBy.ThenBy -> explicit stable
//     index sort; TMPL0 write-only Debug capture dropped. See p2b_v02_parity.md.
//
// OPEN TODO (still need the live NT8 compile/verify loop) -- p2b_v02_parity.md:
//   P2-3  Native z_se bit-parity vs core_v2 _ols_fit_kernel (harness EXPORTED z_se).
//   P2-5  DST-correct America/Chicago RTH / before9 / tod / ts basis (NT8 exchange-local).
//   P2-8  Entry fill semantics -- acts ~180s (3x1m) after the signal minute (consensus
//         settle). Confirm vs the harness "act at bar close T+60" convention.
//   P2-10 Catastrophic stop as a real ExitLongStopMarket for live (not an intrabar poll).
//   P2-12 Warmup / prior-day 5s TAIL + prior-daily profile equivalence vs harness Start.
//   P2-perf ProcessDay is O(N)/minute; optional incremental per-generator port later.
// -----------------------------------------------------------------------------
