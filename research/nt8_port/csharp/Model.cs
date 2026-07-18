// Model.cs -- DayCtx equivalent + shared math (z21 OLS endpoint z, Wilder DMI, TF state).
using System;
using System.Collections.Generic;

namespace Nt8Port
{
    struct Fire
    {
        public int Row; public long Ts; public bool IsLong; public double Value;
        public double PivotAgeMin; public int SigWithLeg; public double Tod;
        public string Det;
    }

    struct PriorDay { public double High, Low, Close, Poc, Vah, Val; public bool HasProfile; }

    class Ctx
    {
        public string Day;
        public int Start, N;
        public long[] Ts;
        public double[] O, H, L, C, V;
        public bool[] Rth, Before9;
        public double[] Tod;
        public double[] Zse;               // may be all-NaN
        public bool HasZse;
        public List<PriorDay> Prior = new List<PriorDay>();

        public const double TICK = 0.25;
        public const int BAR_1M = 12;
        public const int ATR_N = 14;
        public const double ATR_MULT = 4.0;

        public double[] ZzThr;             // per-row ATR(14 1m)x4, index-bucketed by //12
        public int[] PivI;                 // last confirmed pivot row
        public sbyte[] Leg;                // leg direction
        public sbyte[] PivConfirm;         // +1/-1 at confirmation row

        public void BuildDayCtx()
        {
            BuildZzThr();
            BuildZigzag();
        }

        // zz_thr: 1m buckets by ROW INDEX //12 (NOT clock); c1=last,h1=max,l1=min;
        // tr1=max(h1-l1,|h1-pc1|,|l1-pc1|) skipna; atr1=rolling(14,minp14).mean; thr=atr1[i//12]*4
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

        // DayCtx.__init__ streaming zigzag (verbatim)
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
            return new Fire {
                Row = i, Ts = Ts[i], IsLong = isLong, Value = value,
                PivotAgeMin = (i - PivI[i]) * 5.0 / 60.0,
                SigWithLeg = swl, Tod = Tod[i], Det = det
            };
        }

        public IEnumerable<int> RthIdx()
        {
            for (int i = 0; i < N; i++) if (Rth[i] && i >= Start) yield return i;
        }
    }

    static class MathX
    {
        // dsp._z21: 21-bar OLS endpoint z, residual std ddof=2. Returns per-bucket array.
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

        // Wilder-14 DI+/DI- diff (DI+ - DI-), ewm alpha=1/14 adjust=False, on bucketed OHLC.
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
                // note: python np.where on NaN diff -> condition false -> 0.0
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

    // _tf_state: clock-aligned TF buckets + per-bucket V1 quantities + per-row last-closed pos.
    class TfState
    {
        public Buckets B;
        public double[] Z, Vel, Acc, Wick, Vr, Volr, Dmi;
        public int[] RowClosed;

        public static TfState Build(Ctx ctx, long period)
        {
            var b = Buckets.Build(ctx.Ts, ctx.O, ctx.H, ctx.L, ctx.C, ctx.V, period);
            int m = b.Ids.Length;
            var t = new TfState { B = b, RowClosed = b.RowClosed };
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
        public int At(int row) => RowClosed[row];  // -1 if not present
    }
}
