// Pandas.cs -- exact-semantics ports of the pandas/numpy operations the Python
// generators rely on. Every function documents the pandas call it reproduces.
// NaN = double.NaN throughout (matches numpy). No external dependencies.
using System;
using System.Collections.Generic;

namespace Nt8Port
{
    static class Pd
    {
        public static bool Fin(double x) => !double.IsNaN(x) && !double.IsInfinity(x);

        // pandas Series.ewm(alpha=a, adjust=False, ignore_na=False).mean() -- EXACT port of
        // the cython recurrence (aggregations.pyx). For series with only LEADING NaN this
        // reduces to y=(1-a)*prev+a*cur; the difference bites on INTERIOR NaN (e.g. ADX's dx
        // on flat buckets), where ignore_na=False keeps decaying old_wt across the gap and
        // CARRIES the last value at the NaN position (does not emit NaN once seeded).
        public static double[] EwmAlpha(double[] x, double alpha)
        {
            int n = x.Length; var y = new double[n];
            double oldWtFactor = 1.0 - alpha, newWt = alpha;   // adjust=False
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
                oldWt *= oldWtFactor;                          // ignore_na=False: always decay
                if (isObs)
                {
                    if (weighted != cur)
                        weighted = (oldWt * weighted + newWt * cur) / (oldWt + newWt);
                    oldWt = 1.0;                               // adjust=False resets
                }
                y[i] = weighted;                               // carried at NaN positions
            }
            return y;
        }
        public static double[] EwmSpan(double[] x, double span) => EwmAlpha(x, 2.0 / (span + 1.0));
        public static double[] EwmCom(double[] x, double com) => EwmAlpha(x, 1.0 / (1.0 + com));

        // Series.rolling(w, min_periods=minp).mean()
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
        // Series.rolling(w, min_periods=minp).std(ddof)
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
        public static double[] RollMedian(double[] x, int w, int minp)
        {
            int n = x.Length; var y = new double[n]; var buf = new List<double>(w);
            for (int i = 0; i < n; i++)
            {
                if (i < w - 1) { y[i] = double.NaN; continue; }
                buf.Clear();
                for (int j = i - w + 1; j <= i; j++) if (Fin(x[j])) buf.Add(x[j]);
                if (buf.Count < minp) { y[i] = double.NaN; continue; }
                buf.Sort();
                int m = buf.Count;
                y[i] = (m % 2 == 1) ? buf[m / 2] : 0.5 * (buf[m / 2 - 1] + buf[m / 2]);
            }
            return y;
        }
        // Series.diff(): x[i]-x[i-1], first = NaN
        public static double[] Diff(double[] x)
        {
            int n = x.Length; var y = new double[n]; y[0] = double.NaN;
            for (int i = 1; i < n; i++) y[i] = x[i] - x[i - 1];
            return y;
        }
        // Series.shift(k): y[i]=x[i-k], first k = NaN (k>0)
        public static double[] Shift(double[] x, int k)
        {
            int n = x.Length; var y = new double[n];
            for (int i = 0; i < n; i++) y[i] = (i - k >= 0 && i - k < n) ? x[i - k] : double.NaN;
            return y;
        }
    }

    // Clock-aligned OHLCV buckets from the 5s stream: groupby(ts // period).
    // Reproduces the DataFrame.groupby('b') first/max/min/last/sum used across generators.
    class Buckets
    {
        public long[] Ids;                 // unique bucket ids (ts//period), ascending
        public double[] O, H, L, C, V;     // per-bucket aggregates
        public int[] CloseRow;             // bucket pos -> first 5s row of the NEXT bucket (-1 if none)
        public int[] RowClosed;            // per 5s row -> bucket POSITION of (ts//period - 1), else -1
        public Dictionary<long, int> Pos;  // bucket id -> position

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
                long bid = ts[i] / period;   // ts are positive epoch seconds -> floor division ok
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
            // close_row: bucket k -> first row of bucket k+1 (indexed by bucket id-1 in python)
            b.CloseRow = new int[m];
            for (int k = 0; k < m; k++) b.CloseRow[k] = -1;
            for (int k = 1; k < m; k++)
            {
                // python: close_row indexed by (id-1); .get(ids[k]) -> first row of bucket whose id-1==ids[k]
                // i.e. the bucket with id ids[k]+1. Map by position: bucket k's close = first row of next present bucket
                // ONLY if that next bucket's id == ids[k]+1? No: python close_row = Series(first_row.values, index=first_row.index-1)
                // so close_row[bucket_id] exists for bucket_id = (next present bucket id) - 1. .get(ids[k]) returns
                // first row of the present bucket whose id == ids[k]+1, else None.
            }
            // build via id+1 lookup
            for (int k = 0; k < m; k++)
            {
                long wantNext = b.Ids[k] + 1;
                if (b.Pos.TryGetValue(wantNext, out int np2)) b.CloseRow[k] = firstRow[np2];
                else b.CloseRow[k] = -1;
            }
            // row_closed: per row -> position of bucket (ts//period - 1)
            b.RowClosed = new int[n];
            for (int i = 0; i < n; i++)
            {
                long want = ts[i] / period - 1;
                b.RowClosed[i] = b.Pos.TryGetValue(want, out int pp) ? pp : -1;
            }
            return b;
        }
    }
}
