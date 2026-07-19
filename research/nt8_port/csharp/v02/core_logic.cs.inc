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
