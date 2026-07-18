// Tmpl0.cs -- TMPL0 frozen-codebook K-means template stream (verbatim port of
// template_stream_builder.day_events + gen_tmpl0 nearest-centroid assignment).
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace Nt8Port
{
    class Tmpl0
    {
        double[] mean, scale;         // scaler (6)
        double[][] Cs;                // standardized centroids (nTemplates x 6)
        double[] longFrac;           // per template (NaN if null)
        int[] memberCount;
        const double TICK = 0.25;
        const int HURST_N = 30;
        const int MIN_MEMBERS = 20;
        const double MIN_CONV = 0.05;
        static readonly (string, long)[] PERIODS = { ("1m", 60), ("5m", 300), ("15m", 900) };

        public Tmpl0(string codebookPath)
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(codebookPath));
            var root = doc.RootElement;
            mean = ReadArr(root.GetProperty("scaler_mean"));
            scale = ReadArr(root.GetProperty("scaler_scale"));
            var tpl = root.GetProperty("templates");
            int nt = tpl.GetArrayLength();
            Cs = new double[nt][]; longFrac = new double[nt]; memberCount = new int[nt];
            int idx = 0;
            foreach (var t in tpl.EnumerateArray())
            {
                var cen = ReadArr(t.GetProperty("centroid"));
                var cs = new double[6];
                for (int d = 0; d < 6; d++) cs[d] = (cen[d] - mean[d]) / scale[d];
                Cs[idx] = cs;
                memberCount[idx] = t.GetProperty("member_count").GetInt32();
                var lf = t.GetProperty("long_frac");
                longFrac[idx] = lf.ValueKind == JsonValueKind.Null ? double.NaN : lf.GetDouble();
                idx++;
            }
        }
        static double[] ReadArr(JsonElement e)
        {
            var l = new List<double>(); foreach (var v in e.EnumerateArray()) l.Add(v.GetDouble());
            return l.ToArray();
        }

        struct Ev { public int Row; public long Ts; public double PivAge, Tod; public int Leg; public double[] F; public long Tf; }

        // debug capture: per fired event (ts, tid, isLong, standardized features)
        public List<(long ts, int tid, bool isLong, double[] xs, double[] raw)> Debug
            = new List<(long, int, bool, double[], double[])>();

        public List<Fire> Run(Ctx x)
        {
            Debug.Clear();
            var events = new List<Ev>();
            foreach (var (tf, period) in PERIODS)
            {
                var b = Buckets.Build(x.Ts, x.O, x.H, x.L, x.C, x.V, period);
                int m = b.Ids.Length;
                if (m < HURST_N + 2) continue;
                var zAbs = MathX.Z21(b.C);
                var velFeat = new double[m]; velFeat[0] = double.NaN;
                for (int k = 1; k < m; k++) velFeat[k] = Math.Log(1.0 + Math.Abs((b.C[k] - b.C[k - 1]) / TICK));
                double tfFeat = Math.Log2(Math.Max(1, period));
                var (dmi, adx) = WilderDmiAdx(b.H, b.L, b.C);
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
                    var ev = new Ev {
                        Row = r, Ts = x.Ts[r], PivAge = (r - x.PivI[r]) * 5.0 / 60.0,
                        Tod = x.Tod[r], Leg = x.Leg[r], F = fv, Tf = period
                    };
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
                Debug.Add((ev.Ts, tid, isLong, xs, ev.F));
                int swl = ev.Leg != 0 ? ((ev.Leg > 0) == isLong ? 1 : 0) : 0;
                o.Add(new Fire {
                    Row = ev.Row, Ts = ev.Ts, IsLong = isLong, Value = conv,
                    PivotAgeMin = ev.PivAge, SigWithLeg = swl, Tod = ev.Tod, Det = "TMPL0",
                    Tf = ev.Tf
                });
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

        // Wilder-14 DMI diff + ADX (template_stream_builder._wilder_dmi_adx)
        static (double[], double[]) WilderDmiAdx(double[] h, double[] l, double[] c)
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
            var adx = Pd.EwmAlpha(dx, 1.0 / 14.0);
            return (dmi, adx);
        }

        // single-window R/S Hurst H=log(R/S)/log(N), clip[0,1], population std ddof=0
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

        // candlestick codes 0 NONE,1 DOJI,2 HAMMER,3 EBULL,4 EBEAR (priority order)
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

        // geometric codes 0 NONE,1 COMPRESSION,2 WEDGE,3 BREAKDOWN; first 9 forced 0
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
}
