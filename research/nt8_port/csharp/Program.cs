// Program.cs -- P1 C# parity harness entry point. Loads exported inputs, runs the ported
// DayCtx + 22 generators + 22-stream consensus + compact logistic, aggregates to per-1m-bar
// records matching golden_schema, writes csharp/out/<day>.json for parity_check.py compare.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace Nt8Port
{
    class Model
    {
        public string[] Topk; public string[] Cols;
        public double[] Coef, Mu, Sd; public double Threshold;
        public Dictionary<string, int> ColIndex;
        public int NBase;   // number of base+consensus cols before one-hots
    }

    static class Program
    {
        const long CONSENSUS_S = 180;

        static void Main(string[] args)
        {
            string root = args.Length > 0 ? args[0] : ".";
            string data = Path.Combine(root, "harness_data");
            string outDir = Path.Combine(root, "out");
            Directory.CreateDirectory(outDir);
            var model = LoadModel(Path.Combine(data, "_model.json"));
            var tmpl = new Tmpl0(Path.Combine(data, "_tmpl0.json"));
            var files = Directory.GetFiles(data, "*.json.gz").OrderBy(f => f).ToArray();
            Console.WriteLine($"{files.Length} days; topk K={model.Topk.Length}");
            foreach (var f in files)
            {
                var ctx = LoadCtx(f);
                ctx.BuildDayCtx();
                var day = ProcessDay(ctx, model, tmpl);
                WriteDay(Path.Combine(outDir, ctx.Day + ".json"), ctx.Day, day, model);
                if (Environment.GetEnvironmentVariable("TMPL_DEBUG") == "1")
                {
                    var sb = new StringBuilder(); sb.Append('[');
                    for (int di = 0; di < tmpl.Debug.Count; di++)
                    {
                        var e = tmpl.Debug[di]; if (di > 0) sb.Append(',');
                        sb.Append("{\"ts\":").Append(e.ts).Append(",\"tid\":").Append(e.tid)
                          .Append(",\"long\":").Append(e.isLong ? 1 : 0).Append(",\"xs\":[");
                        for (int j = 0; j < 6; j++) { if (j > 0) sb.Append(','); sb.Append(e.xs[j].ToString("R", CultureInfo.InvariantCulture)); }
                        sb.Append("],\"raw\":[");
                        for (int j = 0; j < 6; j++) { if (j > 0) sb.Append(','); sb.Append(e.raw[j].ToString("R", CultureInfo.InvariantCulture)); }
                        sb.Append("]}");
                    }
                    sb.Append(']');
                    File.WriteAllText(Path.Combine(outDir, ctx.Day + ".tmpl0.json"), sb.ToString());
                }
                Console.WriteLine($"  {ctx.Day}: bars={day.Count}");
            }
            Console.WriteLine("done");
        }

        static Model LoadModel(string path)
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var r = doc.RootElement;
            var m = new Model {
                Topk = r.GetProperty("topk").EnumerateArray().Select(x => x.GetString()).ToArray(),
                Cols = r.GetProperty("cols").EnumerateArray().Select(x => x.GetString()).ToArray(),
                Coef = r.GetProperty("coef").EnumerateArray().Select(x => x.GetDouble()).ToArray(),
                Mu = r.GetProperty("mu").EnumerateArray().Select(x => x.GetDouble()).ToArray(),
                Sd = r.GetProperty("sd").EnumerateArray().Select(x => x.GetDouble()).ToArray(),
                Threshold = r.GetProperty("threshold").GetDouble(),
            };
            m.ColIndex = new Dictionary<string, int>();
            for (int i = 0; i < m.Cols.Length; i++) m.ColIndex[m.Cols[i]] = i;
            m.NBase = 5;   // pivot_age_min, sig_with_leg, tod, inter, consensus
            return m;
        }

        static Ctx LoadCtx(string gzPath)
        {
            byte[] raw;
            using (var fs = File.OpenRead(gzPath))
            using (var gz = new GZipStream(fs, CompressionMode.Decompress))
            using (var ms = new MemoryStream()) { gz.CopyTo(ms); raw = ms.ToArray(); }
            using var doc = JsonDocument.Parse(raw);
            var r = doc.RootElement;
            var c = new Ctx {
                Day = r.GetProperty("day").GetString(),
                Start = r.GetProperty("start").GetInt32(),
                N = r.GetProperty("n").GetInt32(),
            };
            c.Ts = r.GetProperty("ts").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            c.O = ReadD(r, "o"); c.H = ReadD(r, "h"); c.L = ReadD(r, "l"); c.C = ReadD(r, "c"); c.V = ReadD(r, "v");
            c.Tod = ReadD(r, "tod");
            c.Rth = r.GetProperty("rth").EnumerateArray().Select(x => x.GetBoolean()).ToArray();
            c.Before9 = r.GetProperty("before9").EnumerateArray().Select(x => x.GetBoolean()).ToArray();
            c.Zse = new double[c.N]; bool hasZ = false;
            int zi = 0;
            foreach (var e in r.GetProperty("zse").EnumerateArray())
            {
                if (e.ValueKind == JsonValueKind.Null) c.Zse[zi] = double.NaN;
                else { c.Zse[zi] = e.GetDouble(); if (!double.IsNaN(c.Zse[zi])) hasZ = true; }
                zi++;
            }
            c.HasZse = hasZ;
            foreach (var pd in r.GetProperty("prior_daily").EnumerateArray())
            {
                var p = new PriorDay {
                    High = pd.GetProperty("high").GetDouble(), Low = pd.GetProperty("low").GetDouble(),
                    Close = pd.GetProperty("close").GetDouble()
                };
                if (pd.TryGetProperty("poc", out var poc)) { p.Poc = poc.GetDouble(); p.HasProfile = true; }
                if (pd.TryGetProperty("vah", out var vah)) p.Vah = vah.GetDouble();
                if (pd.TryGetProperty("val", out var val)) p.Val = val.GetDouble();
                c.Prior.Add(p);
            }
            return c;
        }
        static double[] ReadD(JsonElement r, string k) => r.GetProperty(k).EnumerateArray().Select(x => x.GetDouble()).ToArray();

        // ---- per-bar record ----
        class Bar
        {
            public long BarTs; public Dictionary<string, int> F = new Dictionary<string, int>();
            public string Gov = ""; public int GovDir = 0; public double P = double.NaN;
            public int Entry = 0; public int EntryDir = 0;
        }

        static List<Bar> ProcessDay(Ctx x, Model model, Tmpl0 tmpl)
        {
            // 1. run all 22 generators, tag with a run-order sequence for stable tie-break
            var fires = new List<Fire>();
            void Add(List<Fire> fs) => fires.AddRange(fs);
            Add(Gens.Zigzag(x)); Add(Gens.Orb02(x)); Add(Gens.Vwap03(x)); Add(Gens.Pivot16(x));
            Add(Gens.Round05(x)); Add(Gens.Dow19(x)); Add(Gens.Tunnel20(x)); Add(Gens.Atr09(x));
            Add(Gens.Sar23(x)); Add(Gens.Rsi06(x)); Add(Gens.Macd07(x)); Add(Gens.Renko24(x));
            Add(Gens.Nmpt(x, "FADECALM")); Add(Gens.Nmpt(x, "MTFBRK"));
            Add(Gens.Nmp(x)); Add(Gens.PtrnEngulf(x));
            Add(Gens.CtxEr(x)); Add(Gens.ExitKmdr(x));
            Add(Gens.Nmp9(x, "RIDEAGAINST")); Add(Gens.Nmp9(x, "RIDECALM")); Add(Gens.Nmp9(x, "FADEAGAINST"));
            Add(tmpl.Run(x));

            // 2. consensus (over the full top-K pool) + compact P per fire
            int nf = fires.Count;
            var order = Enumerable.Range(0, nf).OrderBy(i => fires[i].Ts).ThenBy(i => i).ToArray();
            var sorted = order.Select(i => fires[i]).ToList();
            var ts = sorted.Select(f => f.Ts).ToArray();
            var cons = new int[nf];   // indexed in sorted order
            for (int k = 0; k < nf; k++)
            {
                long lo = ts[k] - CONSENSUS_S, hi = ts[k] + CONSENSUS_S;
                int a = LowerBound(ts, lo), b = UpperBound(ts, hi);
                int sameDir = 0, own = 0; bool lng = sorted[k].IsLong; string det = sorted[k].Det;
                for (int j = a; j < b; j++)
                {
                    if (sorted[j].IsLong == lng) { sameDir++; if (sorted[j].Det == det) own++; }
                }
                cons[k] = sameDir - own;
            }
            var P = new double[nf];
            for (int k = 0; k < nf; k++) P[k] = CompactP(sorted[k], cons[k], model);

            // 3. aggregate to per-1m-bar over RTH
            var barMap = new SortedDictionary<long, Bar>();
            for (int i = 0; i < x.N; i++)
                if (x.Rth[i] && i >= x.Start)
                {
                    long T = (x.Ts[i] / 60) * 60;
                    if (!barMap.ContainsKey(T))
                    {
                        var bb = new Bar { BarTs = T };
                        foreach (var d in model.Topk) bb.F[d] = 0;
                        barMap[T] = bb;
                    }
                }
            // fires in ts order -> last direction wins per (bar,det); gov = max P
            for (int k = 0; k < nf; k++)
            {
                long T = (sorted[k].Ts / 60) * 60;
                if (!barMap.TryGetValue(T, out var bar)) continue;   // fire outside RTH bars
                string det = sorted[k].Det;
                bar.F[det] = sorted[k].IsLong ? 1 : -1;
                double p = P[k];
                if (Pd.Fin(p) && (!Pd.Fin(bar.P) || p > bar.P))
                { bar.P = p; bar.Gov = det; bar.GovDir = sorted[k].IsLong ? 1 : -1; }
            }
            foreach (var bar in barMap.Values)
            {
                bar.Entry = (Pd.Fin(bar.P) && bar.P >= model.Threshold) ? 1 : 0;
                bar.EntryDir = bar.Entry == 1 ? bar.GovDir : 0;
            }
            return barMap.Values.ToList();
        }

        static double CompactP(Fire f, int consensus, Model m)
        {
            int nc = m.Cols.Length; double logit = 0;
            for (int ci = 0; ci < nc; ci++)
            {
                string col = m.Cols[ci]; double xv;
                switch (col)
                {
                    case "pivot_age_min": xv = f.PivotAgeMin; break;
                    case "sig_with_leg": xv = f.SigWithLeg; break;
                    case "tod": xv = f.Tod; break;
                    case "inter": xv = f.SigWithLeg * f.PivotAgeMin; break;
                    case "consensus": xv = consensus; break;
                    default: xv = (col == "is_" + f.Det) ? 1.0 : 0.0; break;
                }
                double z = (xv - m.Mu[ci]) / m.Sd[ci];
                logit += z * m.Coef[ci];
            }
            return 1.0 / (1.0 + Math.Exp(-logit));
        }

        static int LowerBound(long[] a, long v)
        {
            int lo = 0, hi = a.Length;
            while (lo < hi) { int mid = (lo + hi) >> 1; if (a[mid] < v) lo = mid + 1; else hi = mid; }
            return lo;
        }
        static int UpperBound(long[] a, long v)
        {
            int lo = 0, hi = a.Length;
            while (lo < hi) { int mid = (lo + hi) >> 1; if (a[mid] <= v) lo = mid + 1; else hi = mid; }
            return lo;
        }

        static void WriteDay(string path, string day, List<Bar> bars, Model model)
        {
            var sb = new StringBuilder();
            sb.Append("{\"day\":\"").Append(day).Append("\",\"bars\":[");
            for (int b = 0; b < bars.Count; b++)
            {
                var bar = bars[b];
                if (b > 0) sb.Append(',');
                sb.Append("{\"bar_ts\":").Append(bar.BarTs);
                foreach (var d in model.Topk) sb.Append(",\"f_").Append(d).Append("\":").Append(bar.F[d]);
                sb.Append(",\"gov_stream\":\"").Append(bar.Gov).Append('"');
                sb.Append(",\"gov_dir\":").Append(bar.GovDir);
                sb.Append(",\"P_compact\":").Append(Pd.Fin(bar.P) ? bar.P.ToString("R", CultureInfo.InvariantCulture) : "null");
                sb.Append(",\"entry\":").Append(bar.Entry);
                sb.Append(",\"entry_dir\":").Append(bar.EntryDir);
                sb.Append('}');
            }
            sb.Append("]}");
            File.WriteAllText(path, sb.ToString());
        }
    }
}
