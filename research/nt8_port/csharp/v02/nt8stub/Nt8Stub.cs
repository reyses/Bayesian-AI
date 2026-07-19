// Nt8Stub.cs -- MINIMAL fake NinjaScript API surface, JUST enough to compile
// docs/nt8/7-EnsembleRunner_v0.2-RC.cs at LangVersion=7.3. This proves the STRATEGY
// WRAPPER (not just the shared core) is C#7.3-clean and type-plausible against the
// NT8 API it calls. It is NOT NinjaTrader and does NOT run -- compile-only (P2-13).
using System;

namespace NinjaTrader.Cbi
{
    public enum MarketPosition { Flat, Long, Short }
    public class Position { public MarketPosition MarketPosition; }
}

namespace NinjaTrader.Data
{
    public enum BarsPeriodType { Second, Minute, Day }
}

namespace NinjaTrader.Gui { public class _g { } }

namespace NinjaTrader.NinjaScript
{
    public enum State { SetDefaults, Configure, DataLoaded, Realtime, Historical, Terminated }
    public enum Calculate { OnBarClose, OnEachTick, OnPriceChange }
    public enum EntryHandling { AllEntries, UniqueEntries }

    [AttributeUsage(AttributeTargets.Property)]
    public class NinjaScriptPropertyAttribute : Attribute { }

    public class PriceSeries { readonly double[] _d = new double[4096]; public double this[int i] { get { return _d[i]; } } }
    public class TimeSeries { public DateTime this[int i] { get { return DateTime.MinValue; } } }
    public class SeriesCollection { readonly PriceSeries _s = new PriceSeries(); public PriceSeries this[int i] { get { return _s; } } }
    public class TimeSeriesCollection { readonly TimeSeries _s = new TimeSeries(); public TimeSeries this[int i] { get { return _s; } } }
}

namespace NinjaTrader.NinjaScript.Strategies
{
    using NinjaTrader.Cbi;
    using NinjaTrader.Data;

    // Base type the strategy derives from. Members mirror the NT8 API the strategy uses.
    public abstract class Strategy
    {
        protected State State;
        protected string Description, Name;
        protected Calculate Calculate;
        protected int EntriesPerDirection, ExitOnSessionCloseSeconds, BarsRequiredToTrade;
        protected EntryHandling EntryHandling;
        protected bool IsExitOnSessionCloseStrategy, IsInstantiatedOnEachOptimizationIteration;
        protected int[] CurrentBars = new int[8];
        protected int CurrentBar;
        protected SeriesCollection Opens = new SeriesCollection(), Highs = new SeriesCollection(),
            Lows = new SeriesCollection(), Closes = new SeriesCollection(), Volumes = new SeriesCollection();
        protected TimeSeriesCollection Times = new TimeSeriesCollection();
        protected Position Position = new Position();
        protected int BarsInProgress;

        protected void AddDataSeries(BarsPeriodType t, int v) { }
        protected void EnterLong(int q, string s) { }
        protected void EnterShort(int q, string s) { }
        protected void ExitLong(string a, string b) { }
        protected void ExitShort(string a, string b) { }

        protected virtual void OnStateChange() { }
        protected virtual void OnBarUpdate() { }
        protected virtual void OnPositionUpdate(Position position, double averagePrice,
                                                int quantity, MarketPosition marketPosition) { }
    }
}
