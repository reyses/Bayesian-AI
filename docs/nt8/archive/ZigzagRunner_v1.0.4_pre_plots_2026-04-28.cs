// =============================================================================
// ZigzagRunner 1.0.4 -- 2026-04-27  (BUGFIX: decouple exit from entry)
// =============================================================================
//
// Pure zigzag pivot-retracement strategy for NinjaTrader 8.
// No CNN, no ML, no Python bridge. Self-contained NinjaScript Strategy.
//
// CHANGELOG 1.0.4 (2026-04-27) -- STRUCTURAL BUGFIX:
//   * BUG: v1.0.0 -> v1.0.3 had NO mid-session exit mechanism. Exits
//     were an ACCIDENTAL SIDE EFFECT of opposite-direction entries:
//        Counter mode: HIGH pivot fires SHORT entry, which (because we
//                      were long from prior LOW pivot) implicitly exits
//                      the long via ExitLong then EnterShort.
//        With mode:    Same alternation hides the missing exit.
//        Always-long:  Every pivot wants LONG. Idempotent guard returns
//                      early. The first-of-session long is held until
//                      EOD force-close. 1.21 trades/day with 18.75 hr
//                      avg time in market = strategy degenerates into
//                      buy-and-hold-each-session.
//        Skip mode:    Same — no exit signal exists.
//
//     User exposed this on 2026-04-27 with OnHighPivot=Long, OnLowPivot=
//     Long. NT8 Strategy Analyzer reported only 5 trades in a 6-day
//     window where ~130 were expected. Diagnosis: the strategy works
//     in counter/with modes only because direction always alternates.
//     The new flexibility in v1.0.3 broke the alternation assumption.
//
//   * FIX: ALWAYS exit current position BEFORE any entry decision.
//     The exit is now a first-class operation, not a side effect.
//
//        if (Position long)  ExitLong(...);
//        if (Position short) ExitShort(...);
//        if (action == Skip) return;       // flat now, stay flat
//        if (action == Long)  EnterLong(...);
//        if (action == Short) EnterShort(...);
//
//   * BACKWARD COMPATIBILITY: Counter and with modes produce IDENTICAL
//     trades to v1.0.3. In those modes direction always alternates, so
//     "exit then enter same direction" never happens. Net behavior
//     unchanged. Always-long / always-short / skip modes now work
//     correctly — every pivot rebalances the position.
//
//   * COMMISSION IMPACT: Always-long mode now generates ~33 trades/day
//     instead of ~1. Commission rises proportionally. The +$269/day
//     observed in v1.0.3 always-long was misleading because it was
//     just regime drift. With proper per-pivot rebalancing, commission
//     friction will be much higher (33 × $1.90 = $63/day in fees alone).
//
//   * The CSV ledger and signal-name semantics are unchanged from
//     v1.0.3. Exit reasons now include "PivotExitLong" / "PivotExitShort"
//     for the new always-exit path; "FlipExitLong" / "FlipExitShort"
//     reasons are removed (replaced by the unconditional pivot exit).
//
// CHANGELOG 1.0.3 (2026-04-27):
//   * NEW: Per-pivot direction control via two PivotAction enum properties.
//     Replaces the v1.0.2 `RideWithTrend` bool with full flexibility:
//
//        OnHighPivot ∈ {Long, Short, Skip}   default = Short (v1.0 baseline)
//        OnLowPivot  ∈ {Long, Short, Skip}   default = Long  (v1.0 baseline)
//
//     Common configurations:
//        High=Short, Low=Long   = v1.0 counter-trend (default)
//        High=Long,  Low=Short  = v1.0.2 RideWithTrend (with-trend)
//        High=Long,  Low=Long   = ALWAYS LONG (= the "flip shorts to long"
//                                 pattern motivated by the 2026-04-27 NT8
//                                 backtest showing shorts have a reliable
//                                 -98%-probability losing edge in MNQ
//                                 March-April regime, while longs profit
//                                 in BOTH counter and with modes)
//        High=Short, Low=Short  = ALWAYS SHORT (mirror of above; would
//                                 work in a downward-biased regime)
//        High=Long,  Low=Skip   = long-on-highs only (= With longs only)
//        High=Skip,  Low=Long   = long-on-lows only  (= Counter longs only)
//
//     Empirical basis from 3/20-4/25/2026 NT8 backtest (R=50, 1s primary):
//        Counter overall: -$275 net, with the breakdown
//          Counter Long:  +$3,135  (LOW pivot -> LONG)
//          Counter Short: -$3,410  (HIGH pivot -> SHORT)
//        With overall:    -$4,081 net, with the breakdown
//          With Long:     +$1,240  (HIGH pivot -> LONG)
//          With Short:    -$5,321  (LOW pivot -> SHORT)
//        Both LONG sides profitable; both SHORT sides losers.
//        Predicted: High=Long + Low=Long = +$3,135 + $1,240 ≈ +$4,375
//        on the same 36-day window (~$122/day).
//
//   * No change to entry/exit timing, EOD, R, position-size hardening,
//     or any other tunable. Direction control is the only delta.
//
// CHANGELOG 1.0.2 (2026-04-26) -- SUPERSEDED BY 1.0.3:
//   * RideWithTrend bool removed. The two-enum approach in 1.0.3 is a
//     strict superset (RideWithTrend=true == OnHighPivot=Long,
//     OnLowPivot=Short).
//
// CHANGELOG 1.0.1 (2026-04-25):
//   * Position-size hardening (no behavior change in normal flow):
//     1. Defensive guard: panic-close if |Position.Quantity| > Contracts
//        at start of any bar update (catches stale/duplicate orders).
//     2. Idempotent entries: skip EnterShort if already short at Contracts;
//        skip EnterLong if already long at Contracts.
//     3. Explicit exit-before-entry: close opposite position with its own
//        ExitLong/ExitShort order BEFORE submitting the new EnterShort/
//        EnterLong. This replaces NT8's combined-flip order (which shows
//        as a 2-contract trade in the order log even though net position
//        is bounded) with two clean 1-contract orders.
//   No change to entry/exit logic, EOD, R, or any tunable parameter.
//
// LOGIC (1.0.3 — direction set by OnHighPivot / OnLowPivot enum properties):
//   1. Zigzag on bar closes with threshold R (points).
//   2. When a pivot is CONFIRMED (price retraces R from current extreme):
//      - HIGH pivot -> action defined by OnHighPivot (Long, Short, or Skip)
//      - LOW  pivot -> action defined by OnLowPivot  (Long, Short, or Skip)
//   3. Past Entry Cutoff UTC: no new entries. Existing position is held
//      until EOD.
//   4. EOD force-close at configurable UTC time.
//
// INSTALLATION:
//   1. This file IS the live copy in NT8 Strategies/ folder.
//   2. In NT8: Tools > NinjaScript Editor > Compile (F5).
//   3. Apply to MNQ 1-minute chart, Sim101 account, Slippage = 1 tick.
//
// =============================================================================

#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ZigzagRunner : Strategy
    {
        // ── Settings ─────────────────────────────────────────────────────

        [NinjaScriptProperty]
        [Display(Name = "R (points)", Description = "Zigzag retracement threshold in price points (30 = MNQ $60)", Order = 1, GroupName = "Zigzag")]
        public double RPoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Contracts", Description = "Contracts per trade", Order = 2, GroupName = "Zigzag")]
        public int Contracts { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EOD Hour UTC", Description = "Force close hour (UTC, 0-23). 20 = before NYSE close", Order = 1, GroupName = "Schedule")]
        public int EodHourUtc { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EOD Minute UTC", Description = "Force close minute (0-59)", Order = 2, GroupName = "Schedule")]
        public int EodMinuteUtc { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Entry Cutoff Hour UTC", Description = "No new entries after this time (UTC)", Order = 3, GroupName = "Schedule")]
        public int EntryCutoffHourUtc { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Entry Cutoff Minute UTC", Description = "No new entries after this minute (0-59)", Order = 4, GroupName = "Schedule")]
        public int EntryCutoffMinuteUtc { get; set; }

        // ── Per-pivot direction control (v1.0.3) ─────────────────────────
        // Two independent enum properties decide what each pivot type does.
        // Default (Short on highs, Long on lows) = original v1.0 counter-
        // trend behavior. Set both to Long for "always long, flip shorts".

        [NinjaScriptProperty]
        [Display(Name = "On High Pivot", Description = "Action when a HIGH pivot is confirmed. Default Short = v1.0 counter-trend. Set to Long for trend-following entry on highs (with-mode longs). Set to Skip to disable entries on highs.", Order = 1, GroupName = "Direction")]
        public PivotAction_v10 OnHighPivot { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "On Low Pivot", Description = "Action when a LOW pivot is confirmed. Default Long = v1.0 counter-trend. Set to Short for trend-following entry on lows (with-mode shorts). Set to Skip to disable entries on lows.", Order = 2, GroupName = "Direction")]
        public PivotAction_v10 OnLowPivot { get; set; }

        // ── Version ──────────────────────────────────────────────────────
        private const string VERSION = "1.0.4";

        // ── Zigzag state ─────────────────────────────────────────────────
        // direction: 0 = undefined (before first pivot), +1 = up leg, -1 = down leg
        private int direction;
        private double extremePrice;
        private int extremeBarIdx;
        private int lastPivotDir;      // +1 = high pivot (just formed), -1 = low pivot; 0 = none yet
        private double lastPivotPrice;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name                             = "ZigzagRunner";
                Description                      = "Zigzag pivot-retracement strategy (pure rule, no ML). v" + VERSION;
                Calculate                        = Calculate.OnBarClose;
                EntriesPerDirection              = 1;
                EntryHandling                    = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy     = true;
                ExitOnSessionCloseSeconds        = 30;
                IsFillLimitOnTouch               = false;
                MaximumBarsLookBack              = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution              = OrderFillResolution.Standard;
                Slippage                         = 0;
                StartBehavior                    = StartBehavior.WaitUntilFlat;
                TimeInForce                      = TimeInForce.Gtc;
                TraceOrders                      = false;
                RealtimeErrorHandling            = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling               = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade              = 2;

                // Defaults matching Python backtest sweet spot
                RPoints                          = 30.0;
                Contracts                        = 1;
                EodHourUtc                       = 20;
                EodMinuteUtc                     = 55;
                EntryCutoffHourUtc               = 20;
                EntryCutoffMinuteUtc             = 30;
                OnHighPivot                      = PivotAction_v10.Short;  // v1.0 baseline (counter-trend)
                OnLowPivot                       = PivotAction_v10.Long;   // v1.0 baseline (counter-trend)
            }
            else if (State == State.Configure)
            {
                direction       = 0;
                extremePrice    = double.NaN;
                extremeBarIdx   = -1;
                lastPivotDir    = 0;
                lastPivotPrice  = double.NaN;
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < BarsRequiredToTrade) return;

            // ─── 1.0.1 SAFETY GUARD ──────────────────────────────────────
            // If position size somehow exceeded Contracts (stale order, manual
            // intervention, NT8 framework bug), panic-close immediately.
            // This is a "should never happen" backstop.
            if (Math.Abs(Position.Quantity) > Contracts)
            {
                Print("ZigzagRunner SAFETY: Position.Quantity=" + Position.Quantity +
                      " exceeds Contracts=" + Contracts + ", panic-closing.");
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong(Position.Quantity, "SafetyPanicLong", "");
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort(Position.Quantity, "SafetyPanicShort", "");
                return;
            }

            double c = Close[0];
            DateTime barUtc = Time[0].ToUniversalTime();
            int minsOfDay      = barUtc.Hour * 60 + barUtc.Minute;
            int eodMins        = EodHourUtc * 60 + EodMinuteUtc;
            int entryCutMins   = EntryCutoffHourUtc * 60 + EntryCutoffMinuteUtc;

            // ─── EOD force-close ───────────────────────────────────────
            if (minsOfDay >= eodMins)
            {
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong(Position.Quantity, "EodExitLong", "");
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort(Position.Quantity, "EodExitShort", "");
                return;
            }

            // ─── Initialize extreme on first bar ───────────────────────
            if (double.IsNaN(extremePrice))
            {
                extremePrice  = c;
                extremeBarIdx = CurrentBar;
                return;
            }

            // ─── Zigzag state machine ──────────────────────────────────
            bool pivotConfirmed = false;
            int newPivotDir     = 0;   // +1 = high pivot, -1 = low pivot

            if (direction == 0)
            {
                // No direction yet — first R-retracement defines the first leg
                if (c - extremePrice >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = -1;           // extreme was a LOW pivot
                    lastPivotPrice = extremePrice;
                    direction      = +1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBar;
                }
                else if (extremePrice - c >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = +1;           // extreme was a HIGH pivot
                    lastPivotPrice = extremePrice;
                    direction      = -1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBar;
                }
            }
            else if (direction == +1)
            {
                // In an UP leg — watch for new highs, otherwise R-retracement confirms HIGH pivot
                if (c > extremePrice)
                {
                    extremePrice  = c;
                    extremeBarIdx = CurrentBar;
                }
                else if (extremePrice - c >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = +1;           // high pivot
                    lastPivotPrice = extremePrice;
                    direction      = -1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBar;
                }
            }
            else  // direction == -1 (down leg)
            {
                if (c < extremePrice)
                {
                    extremePrice  = c;
                    extremeBarIdx = CurrentBar;
                }
                else if (c - extremePrice >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = -1;           // low pivot
                    lastPivotPrice = extremePrice;
                    direction      = +1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBar;
                }
            }

            if (!pivotConfirmed) return;
            lastPivotDir = newPivotDir;

            // ─── Entry cutoff (v1.0 behavior: skip entry, no exit) ─────
            // Note: v1.1+ adds a reverse-exit here. v1.0 just holds.
            if (minsOfDay >= entryCutMins) return;

            // ─── 1.0.4: STRUCTURAL FIX — decouple exit from entry ────────
            // Every pivot ALWAYS exits the current position FIRST. This
            // closes a v1.0.0-1.0.3 bug: exits were an accidental side
            // effect of opposite-direction entries. In counter/with modes
            // direction always alternated, so the exit happened naturally
            // on every pivot. In always-long / always-short / skip modes,
            // there was no opposite-direction trigger, so the position
            // stayed open until EOD force-close. Trades degenerated to
            // ~1/day overnight holds.
            //
            // Counter and with modes are byte-equivalent to v1.0.3 because
            // direction always alternates: ExitLong then EnterShort vs.
            // (idempotent skip + ExitLong + EnterShort) produce the same
            // single round-trip. The new logic is a strict superset.
            //
            // Always-long / always-short / skip modes now correctly fire
            // ~33 trades/day with proper per-pivot exits.
            PivotAction_v10 action;
            string pivotLabel;
            if (newPivotDir == +1)
            {
                action     = OnHighPivot;
                pivotLabel = "HighPivot";
            }
            else
            {
                action     = OnLowPivot;
                pivotLabel = "LowPivot";
            }

            // ALWAYS exit current position first (whatever direction it is).
            if (Position.MarketPosition == MarketPosition.Long)
                ExitLong(Position.Quantity, "PivotExitLong", "");
            else if (Position.MarketPosition == MarketPosition.Short)
                ExitShort(Position.Quantity, "PivotExitShort", "");

            // Skip = stay flat after the exit above. No new entry fires.
            if (action == PivotAction_v10.Skip)
                return;

            if (action == PivotAction_v10.Short)
                EnterShort(Contracts, "ShortAt" + pivotLabel);
            else  // PivotAction_v10.Long
                EnterLong(Contracts, "LongAt" + pivotLabel);
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // PivotAction_v10 (v1.0.3, 2026-04-27)
    // ─────────────────────────────────────────────────────────────────────
    // Per-pivot-type direction action. Public + version-suffixed name so
    // it can coexist with similar enums in v1.1/v1.2/etc. without colliding
    // when NT8 compiles the entire Strategies/ folder into one assembly.
    // ═════════════════════════════════════════════════════════════════════
    public enum PivotAction_v10
    {
        Long,    // EnterLong on this pivot type
        Short,   // EnterShort on this pivot type
        Skip,    // No entry on this pivot type
    }
}
