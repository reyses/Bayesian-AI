//
// 7-EnsembleRunner_v0.1-RC.cs
// -----------------------------------------------------------------------------
// NinjaTrader 8 STRATEGY: Ensemble entry combiner + R-trigger ride-only exit.
//
// ORIGIN (2026-07-18, doc 133 / research/nt8_port P2): native NinjaScript
// packaging of the Architecture-B entry engine that was validated bar-by-bar in
// the C# parity harness (research/nt8_port/csharp). The harness reproduces the
// Python reference decider to BIT-EXACT parity over 20 regime-diverse reference
// days:
//   * 22 top-K stream fire-states : 100.000% (178,640 / 178,640 cells)
//   * compact combiner P          : max |dP| = 2.22e-16
//   * entry decision @ threshold  : 100.000% (8,120 / 8,120 bars)
//   * R-trigger zigzag leg+pivot  : 100.000%, pivot age/price bit-exact (0.0)
// (see research/nt8_port/reports/p1_parity.md + p2_report.md)
//
// WHAT THIS IS: a mechanical manager (Architecture B, doc 129) -- NO cut logic.
//   Entry  : pooled 22-stream logistic combiner P >= frozen top-decile threshold
//            (0.713983). Side = the governing (max-P) stream's direction.
//   Exit   : R-trigger REVERSAL ONLY (ride-only, doc 107). A confirmed zigzag
//            pivot AGAINST the open position closes it. No fixed TP, no MFE cut,
//            no trailing stop -- the leg rides until the structure turns.
//   Sizing : fixed 1 contract.
//   Guards : optional catastrophic stop (default OFF in SIM, present for live);
//            session flatten at 15:55 CT.
//
// SOURCE OF TRUTH: the 22 generators, the TMPL0 codebook stream, the consensus
// window, the compact logistic, and the R-trigger state machine are the SAME
// code validated in research/nt8_port/csharp/{Gens,Tmpl0,Model,Program}.cs. This
// file is the STREAMING (OnBarUpdate) adaptation of that batch harness. Every
// point where the streaming port still needs the live NT8 compile/verify loop is
// marked  // TODO(P2-#N)  and enumerated in the CHANGELOG + p2_report.md.
//
// DEPLOY GATE: this is an -RC. It has NOT been NT8-compiled and NOTHING has been
// copied to Documents/NinjaTrader 8/bin/Custom/Strategies/. Per the house
// deploy-gate policy, promotion requires explicit per-revision user approval.
//
// VERSION: 0.1-RC
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
#endregion

// NOTE(P2-compat): the reference harness targets .NET 10 / C# 12 (tuples with
// named members, target-typed `new`, switch expressions). NinjaTrader 8 compiles
// NinjaScript against .NET Framework 4.8 / an older Roslyn. This draft deliberately
// avoids net10-only syntax so it is compile-plausible under NT8; when porting the
// remaining generator bodies from Gens.cs, down-level the same way. // TODO(P2-13)
namespace NinjaTrader.NinjaScript.Strategies
{
    public class EnsembleRunner_v01 : Strategy
    {
        private const string VERSION = "0.1-RC";

        // ---- frozen decision constants (research/nt8_port, DO NOT re-tune here) ----
        // Origin: golden_manifest.json + _model.json (2026-07-18 P2 build).
        private const double TICK = 0.25;                    // MNQ tick size (points)
        private const double ENTRY_THRESHOLD = 0.7139834155227371; // 90th pct 2024 compact-P
        private const int    CONSENSUS_WINDOW_SEC = 180;     // +-180s same-dir co-fire window
        private const int    TOPK = 22;                      // top-K streams (80% of stream mass)

        // ---- R-trigger zigzag constants (training/strategies/zigzag.py, verbatim) ----
        private const int    ZZ_MIN_BARS_5S = 36;            // ZigzagStrategy.MIN_BARS_5S_DEFAULT
        private const int    ZZ_ATR_PERIOD  = 14;            // ATR(14) of 1m bars
        private const double ZZ_ATR_MULT    = 4.0;           // R = ATR14 x 4 (points), open-anchored
        private const int    ZZ_MIN_R_TICKS = 4;             // floor: R = max(4, round(R_pts/TICK))

        // ---- z_se native-derivation constants (core_v2 statistical_field_engine) ----
        private const int    ZSE_WINDOW_1M = 15;             // N_BASE[1m]=15 -> L3_1m_z_se_15
        private const int    ZSE_OLS_DDOF  = 2;              // residual std ddof=2 (endpoint OLS)

        // ---- session calendar (America/Chicago RTH; NT8 exchange-local) ----
        private const int RTH_OPEN_HH = 8,  RTH_OPEN_MM = 30;   // 08:30 CT
        private const int RTH_CLOSE_HH = 15, RTH_CLOSE_MM = 15;  // 15:15 CT (bar-close gate)

        // ---- secondary-series indices (assigned in Configure) ----
        private const int BIP_5S = 0;   // primary chart series MUST be 5s
        private const int BIP_1M = 1;   // z_se OLS + zz_thr ATR(14) live here

        // ---- streaming engine state ----
        private EnsembleEngine engine;      // 22 generators + consensus + combiner + TMPL0
        private RTriggerZigzag zigzag;      // native R-trigger (validated port)
        private double curR_points = double.NaN;   // per-day open-anchored R (points)
        private bool   rLocked = false;            // R frozen at first RTH 5s bar of the day
        private int    openDir = 0;                // +1 long / -1 short / 0 flat
        private double openEntryPrice = double.NaN;
        private bool   tradeAllowedToday = true;   // cleared by session flatten guard
        private DateTime sessionDate = DateTime.MinValue;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Ensemble entry combiner (22-stream logistic P >= 0.713983) with " +
                    "R-trigger ride-only reversal exit. Architecture B, mechanical manager, no cut " +
                    "logic. Bit-exact port of research/nt8_port parity harness (100% fire/entry/pivot " +
                    "parity, 20 reference days). RC -- NOT deploy-approved.";
                Name = "EnsembleRunner_v0.1-RC";
                Calculate = Calculate.OnBarClose;          // closed-bar semantics (no lookahead)
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 60;
                BarsRequiredToTrade = WarmupBars;          // see property (longest lookback + tail)
                IsInstantiatedOnEachOptimizationIteration = true;

                Quantity = 1;                              // fixed 1 contract
                EnableCatastrophicStop = false;            // OFF in SIM by default
                CatastrophicStopPoints = 200;              // present for live; ignored while OFF
                SessionFlattenHH = 15;                     // 15:55 CT flatten guard
                SessionFlattenMM = 55;
                ZSeMode = ZSeFeedMode.Native;              // native OLS z_se by default
                WarmupBars = 5000;                         // ~ prior-day tail + longest generator win
            }
            else if (State == State.Configure)
            {
                // Primary series (index 0) is whatever chart the strategy is applied to.
                // We REQUIRE it to be 5-second bars (the substrate the harness streams).
                // TODO(P2-1): assert/verify BarsPeriod == 5s at DataLoaded; reject otherwise.
                AddDataSeries(BarsPeriodType.Minute, 1);   // BIP_1M = 1 : z_se + zz_thr ATR14
            }
            else if (State == State.DataLoaded)
            {
                engine = new EnsembleEngine(ENTRY_THRESHOLD, CONSENSUS_WINDOW_SEC);
                zigzag = new RTriggerZigzag(ZZ_MIN_BARS_5S);
                // TODO(P2-2): load & embed the frozen combiner model (_model.json: 27 cols,
                //   coef, mu, sd, topk order) and the TMPL0 codebook (_tmpl0.json) as compiled
                //   resources; verify byte-identity vs research/nt8_port/csharp/harness_data/.
                // TODO(P2-4): wire TMPL0 nearest-centroid + the P2 same-bar tie rule
                //   (highest-TF wins; tie->larger conviction; still tied->hold prior=0).
            }
        }

        protected override void OnBarUpdate()
        {
            // ---------- 1-minute series: z_se + zz_thr(ATR14x4) ----------
            if (BarsInProgress == BIP_1M)
            {
                if (CurrentBars[BIP_1M] < ZZ_ATR_PERIOD) return;
                // zz_thr (points) = ATR(14) of 1m bars * 4. Open-anchored R is LOCKED at the
                // first RTH 5s bar of the day (below); this keeps the rolling value ready.
                double atr14 = Atr1m(ZZ_ATR_PERIOD);           // TODO(P2-6): pandas-exact rolling ATR
                curR_points = atr14 * ZZ_ATR_MULT;
                if (ZSeMode == ZSeFeedMode.Native)
                {
                    // z_se = (close - RM_close)/SE_close, endpoint OLS window 15, residual std ddof=2.
                    // Identical formula family to the harness MathX.Z21 (window 21) -> portable.
                    // TODO(P2-3): bit-parity check vs core_v2 statistical_field_engine._ols_fit_kernel
                    //   before live (P1 EXPORTED z_se as an external input rather than deriving it).
                    engine.PushZse(OlsEndpointZ(BIP_1M, ZSE_WINDOW_1M, ZSE_OLS_DDOF), Times[BIP_1M][0]);
                }
                else
                {
                    // TODO(P2-3b): file-feed path -- read L3_1m_z_se_15 from an exported side file
                    //   keyed by bar timestamp. Fallback if the native derivation fails parity.
                }
                return;
            }

            // ---------- 5-second primary series: the decision path ----------
            if (BarsInProgress != BIP_5S) return;
            if (CurrentBar < BarsRequiredToTrade) return;

            DateTime t = Times[BIP_5S][0];
            RollSession(t);                                    // new-day reset of R lock + flags
            bool inRth = IsRth(t);                             // TODO(P2-5): DST-correct CT session gate

            // Lock the per-day R at the first RTH 5s bar (causal open-anchored ATR).
            if (inRth && !rLocked && !double.IsNaN(curR_points))
            {
                int rTicks = Math.Max(ZZ_MIN_R_TICKS, (int)Math.Round(curR_points / TICK,
                                        MidpointRounding.ToEven));
                zigzag.SetR(rTicks);
                rLocked = true;
            }

            // Advance the R-trigger on EVERY 5s close (needs the full stream, incl. pre-RTH).
            int zzConfirm = zigzag.Update(Closes[BIP_5S][0] / TICK);   // +1/-1 on a confirmed pivot

            // ---- EXIT first: R-trigger reversal against the open position (ride-only) ----
            if (openDir != 0 && zzConfirm != 0 && zzConfirm == -openDir)
            {
                FlattenPosition("RTriggerReversal");
            }

            // ---- catastrophic stop (live only; OFF in SIM by default) ----
            if (EnableCatastrophicStop && openDir != 0)
            {
                double adverse = openDir > 0
                    ? (openEntryPrice - Lows[BIP_5S][0])
                    : (Highs[BIP_5S][0] - openEntryPrice);
                if (adverse >= CatastrophicStopPoints)
                    FlattenPosition("CatastrophicStop");
            }

            // ---- session flatten guard (15:55 CT): flatten + block new entries ----
            if (AtOrAfter(t, SessionFlattenHH, SessionFlattenMM))
            {
                if (openDir != 0) FlattenPosition("SessionFlatten");
                tradeAllowedToday = false;
            }

            if (!inRth || !tradeAllowedToday) return;

            // ---- ENTRY: run the ensemble on this closed 5s bar, act at threshold ----
            // The engine reproduces the harness per-bar aggregation: run 22 generators, map
            // fires to this minute, compute per-fire consensus + compact P, take the governing
            // (max-P) fire. Entry iff gov P >= threshold.
            // TODO(P2-1): the 22 generator bodies + TMPL0 are ported here from the validated
            //   harness Gens.cs/Tmpl0.cs into incremental streaming form (per-bar, not batch).
            // TODO(P2-7): consensus window uses a rolling +-180s fire buffer (streaming form of
            //   golden_vector_gen.day_consensus).
            EnsembleDecision d = engine.OnFiveSecClose(
                Times[BIP_5S][0], Opens[BIP_5S][0], Highs[BIP_5S][0],
                Lows[BIP_5S][0], Closes[BIP_5S][0], Volumes[BIP_5S][0],
                zigzag.Leg, zigzag.PivotAgeMinutes(CurrentBar));

            if (openDir == 0 && d.Entry && d.Dir != 0)
            {
                // TODO(P2-8): confirm market-on-close-bar fill semantics vs the harness "act at
                //   bar close T+60" convention. EntriesPerDirection=1 caps to one open leg.
                if (d.Dir > 0) EnterLong(Quantity, "Long");
                else           EnterShort(Quantity, "Short");
            }
        }

        // Track fills to know our open direction/price for the ride-only exit + stop.
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
            if (Position.MarketPosition == MarketPosition.Long)  ExitLong("X_" + reason, "Long");
            else if (Position.MarketPosition == MarketPosition.Short) ExitShort("X_" + reason, "Short");
            openDir = 0;
        }

        private void RollSession(DateTime t)
        {
            if (t.Date != sessionDate)
            {
                sessionDate = t.Date;
                rLocked = false;
                tradeAllowedToday = true;
                zigzag.NewDayResetRLock();   // R re-locks at the next RTH 5s bar
                // NOTE: the R-trigger STATE STREAM is continuous across days (matches the harness,
                // which streams the full 5s series incl. the prior-day tail). Only the R VALUE
                // re-locks per day. // TODO(P2-12): confirm warmup/tail equivalence vs harness TAIL.
            }
        }

        private bool IsRth(DateTime t)
        {
            int mins = t.Hour * 60 + t.Minute;
            return mins >= (RTH_OPEN_HH * 60 + RTH_OPEN_MM) &&
                   mins <= (RTH_CLOSE_HH * 60 + RTH_CLOSE_MM);
        }

        private bool AtOrAfter(DateTime t, int hh, int mm)
            => (t.Hour * 60 + t.Minute) >= (hh * 60 + mm);

        // Rolling ATR(14) on the 1m series. // TODO(P2-6): match the harness zz_thr basis exactly
        // (index-bucketed //12 tr = max(h-l,|h-pc|,|l-pc|), rolling(14).mean).
        private double Atr1m(int period)
        {
            double sum = 0;
            for (int k = 0; k < period; k++)
            {
                double h = Highs[BIP_1M][k], l = Lows[BIP_1M][k];
                double pc = (k + 1 <= CurrentBars[BIP_1M]) ? Closes[BIP_1M][k + 1] : Closes[BIP_1M][k];
                double tr = Math.Max(h - l, Math.Max(Math.Abs(h - pc), Math.Abs(l - pc)));
                sum += tr;
            }
            return sum / period;
        }

        // Endpoint OLS z: (y_end - fit_end) / residual_std(ddof). Window on the BIP series.
        // Mirrors core_v2 _ols_fit_kernel + harness MathX.Z21. // TODO(P2-3)
        private double OlsEndpointZ(int bip, int window, int ddof)
        {
            if (CurrentBars[bip] < window - 1) return 0.0;
            double xm = (window - 1) / 2.0, xv = 0, ym = 0;
            for (int k = 0; k < window; k++) { double dx = k - xm; xv += dx * dx; ym += Closes[bip][k]; }
            ym /= window;
            double num = 0;
            // Closes[bip][0] is the newest bar; map k=window-1 -> endpoint.
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
            return sd > 1e-10 ? (Closes[bip][0] - fitLast) / sd : 0.0;
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
        [Display(Name = "Warmup bars (5s)", Order = 1, GroupName = "4. Warmup")]
        public int WarmupBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "z_se source", Order = 1, GroupName = "5. Features")]
        public ZSeFeedMode ZSeMode { get; set; }
        #endregion
    }

    public enum ZSeFeedMode { Native, FileFeed }

    // ===========================================================================
    // R-trigger zigzag -- streaming form of the validated port
    // (research/nt8_port/csharp/Program.ZigzagRTrigger, itself a verbatim port of
    //  training/strategies/zigzag.py::ZigzagStrategy). Prices in TICKS.
    // Update() returns +1/-1 on a CONFIRMED pivot (leg flip), else 0.
    // ===========================================================================
    public class RTriggerZigzag
    {
        private readonly int minBars5s;
        private int minRevTicks = 0;        // R in ticks; set at the first RTH 5s bar
        private bool rSet = false;
        private int d = 0;                  // leg direction (0/+1/-1)
        private double ext, firstClose;
        private int i = 0, extBar = 0;
        private int lastPivBar = 0; private double lastPivPx = double.NaN;
        private bool seeded = false;

        public RTriggerZigzag(int minBars) { minBars5s = minBars; }
        public int Leg { get { return d; } }
        public void SetR(int rTicks) { minRevTicks = rTicks; rSet = true; }
        public void NewDayResetRLock() { /* R re-locks; STATE stream persists (harness parity) */ }
        public double PivotAgeMinutes(int curBar) { return (i - lastPivBar) * 5.0 / 60.0; }

        public int Update(double priceTicks)
        {
            if (!seeded) { ext = firstClose = lastPivPx = priceTicks; extBar = 0; seeded = true; i = 0; return 0; }
            i++;
            if (!rSet) return 0;                        // no signal until R locked (RTH open)
            double p = priceTicks; int f = 0;
            if (d == 0)
            {
                if (p > ext) { ext = p; extBar = i; }
                if (p < firstClose && (firstClose - p) >= minRevTicks) { d = -1; ext = p; extBar = i; f = -1; }
                else if (p > firstClose && (p - firstClose) >= minRevTicks) { d = 1; ext = p; extBar = i; f = 1; }
                if (f != 0) { lastPivBar = i; lastPivPx = firstClose; }
            }
            else if (d == 1)
            {
                if (p >= ext) { ext = p; extBar = i; }
                else if ((ext - p) >= minRevTicks && (i - extBar) >= minBars5s)
                { lastPivBar = extBar; lastPivPx = ext; d = -1; ext = p; extBar = i; f = -1; }
            }
            else // d == -1
            {
                if (p <= ext) { ext = p; extBar = i; }
                else if ((p - ext) >= minRevTicks && (i - extBar) >= minBars5s)
                { lastPivBar = extBar; lastPivPx = ext; d = 1; ext = p; extBar = i; f = 1; }
            }
            return f;
        }
    }

    // ===========================================================================
    // Ensemble engine -- the 22 generators + consensus + compact logistic + TMPL0.
    // This is the STREAMING (OnBarUpdate) adaptation of the batch parity harness
    // (research/nt8_port/csharp/{Gens,Tmpl0,Program}.cs). The math is VALIDATED
    // there at 100% parity; this class is the port boundary.
    // ===========================================================================
    public struct EnsembleDecision { public bool Entry; public int Dir; public double P; public string Gov; }

    public class EnsembleEngine
    {
        private readonly double threshold;
        private readonly int consensusWindowSec;

        public EnsembleEngine(double thr, int consWinSec)
        {
            threshold = thr; consensusWindowSec = consWinSec;
            // TODO(P2-2): load frozen model (27 cols, coef, mu, sd, topk order) + TMPL0 codebook.
        }

        public void PushZse(double zse, DateTime t)
        {
            // TODO(P2-3): feed L3_1m_z_se_15 to the NMP / NMP9-head generators (streaming buffer).
        }

        // Called on each closed 5s RTH bar. Returns the governing entry decision.
        public EnsembleDecision OnFiveSecClose(DateTime t, double o, double h, double l,
                                               double c, double v, int leg, double pivotAgeMin)
        {
            // TODO(P2-1): run the 22 top-K streaming generators (ports of Gens.cs):
            //   RSI06, MACD07, EXITKMDR, TMPL0, ZIGZAG, ATR09, NMP, DOW19, NMP9RIDEAGAINST,
            //   ROUND05, NMPTFADECALM, RENKO24, ORB02, VWAP03, CTXER, PIVOT16, SAR23,
            //   PTRNENGULF, NMP9RIDECALM, NMPTMTFBRK, TUNNEL20, NMP9FADEAGAINST.
            // TODO(P2-4): TMPL0 nearest-centroid fire(s) + P2 same-bar tie rule.
            // TODO(P2-7): per-fire consensus over the rolling +-180s same-direction fire buffer.
            // TODO(P2-2): compact logistic P per fire; governing = argmax P; entry iff P>=threshold.
            return new EnsembleDecision { Entry = false, Dir = 0, P = double.NaN, Gov = "" };
        }
    }
}

// -----------------------------------------------------------------------------
// CHANGELOG
//   v0.1-RC (2026-07-18, doc 133 P2): initial NinjaScript draft. Structural
//     skeleton compile-plausible under NT8 (.NET 4.8); fully-implemented parts =
//     entry gate wiring, R-trigger ride-only exit, catastrophic stop, session
//     flatten guard, 1-contract sizing, native z_se endpoint-OLS, per-day
//     open-anchored R lock. The 22-generator + consensus + logistic + TMPL0
//     bodies are the VALIDATED harness code (research/nt8_port/csharp), ported
//     as a streaming engine boundary. NOT NT8-compiled; NOTHING deployed.
//
// OPEN TODO (needs the live NT8 compile/verify loop) -- see p2_report.md:
//   P2-1  Port the 22 generator bodies (Gens.cs) into incremental streaming form.
//   P2-2  Embed/verify the frozen combiner model (_model.json) + entry threshold.
//   P2-3  Native z_se bit-parity vs core_v2 _ols_fit_kernel (P1 exported it).
//   P2-3b File-feed z_se fallback path.
//   P2-4  TMPL0 codebook load + nearest-centroid + P2 same-bar tie rule.
//   P2-5  DST-correct America/Chicago RTH / tod / before9 session gate.
//   P2-6  zz_thr ATR(14 1m)x4 basis identical to harness (index //12 buckets).
//   P2-7  Consensus rolling +-180s same-direction fire buffer (streaming form).
//   P2-8  Entry fill semantics vs harness "act at bar close T+60".
//   P2-9  R-trigger exit: confirmed pivot AGAINST open leg closes it (ride-only).
//   P2-10 Catastrophic stop as a real ExitLongStopMarket for live (not intrabar poll).
//   P2-11 Session flatten 15:55 CT + block re-entry until next session.
//   P2-12 Warmup/prior-day-tail equivalence vs harness TAIL context.
//   P2-13 Down-level net10/C#12 syntax to NT8 (.NET 4.8 / older Roslyn).
// -----------------------------------------------------------------------------
