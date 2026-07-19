// ShimMain.cs -- V02 parity shim. Drives the SHARED v0.2 core classes
// (EnsembleCoreV02.gen.cs, generated from the single canonical region that also
// ships inside docs/nt8/7-EnsembleRunner_v0.2-RC.cs) over the golden 20 days and
// writes out_v02/<day>.json in the EXACT harness format. p2b_verify.py then diffs
// out_v02/ against the harness out_baseline/ (which is 100% vs golden) -> if
// byte-identical, the ported v0.2 core == golden, and the down-level (C#7.3,
// no ValueTuple, no Math.Log2, embedded constants) is proven parity-neutral.
//
// Deliberately compiled at LangVersion=7.3 (see V02ParityShim.csproj) so a green
// build is also the P2-13 down-level proof for the whole shared region + this IO.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.Json;
using EnsembleV02Core;

namespace EnsembleV02Shim
{
    static class ShimMain
    {
        static int Main(string[] args)
        {
            string root = args.Length > 0 ? args[0] : ".";
            string data = Path.Combine(root, "harness_data");
            string outDir = Path.Combine(root, "out_v02");
            Directory.CreateDirectory(outDir);
            var tmpl = new Tmpl0();   // embedded codebook (no JSON load)
            var files = Directory.GetFiles(data, "*.json.gz").OrderBy(f => f).ToArray();
            Console.WriteLine(files.Length + " days; topk K=" + ModelData.Topk.Length
                + "; templates=" + Tmpl0Data.NTemplates);
            int totBars = 0;
            foreach (var f in files)
            {
                var ctx = LoadCtx(f);
                ctx.BuildDayCtx();
                var day = Core.ProcessDay(ctx, tmpl);
                WriteDay(Path.Combine(outDir, ctx.Day + ".json"), ctx.Day, day);
                totBars += day.Count;
                Console.WriteLine("  " + ctx.Day + ": bars=" + day.Count);
            }
            Console.WriteLine("done; totalBars=" + totBars);
            return 0;
        }

        static Ctx LoadCtx(string gzPath)
        {
            byte[] raw;
            using (var fs = File.OpenRead(gzPath))
            using (var gz = new GZipStream(fs, CompressionMode.Decompress))
            using (var ms = new MemoryStream()) { gz.CopyTo(ms); raw = ms.ToArray(); }
            using (var doc = JsonDocument.Parse(raw))
            {
                var r = doc.RootElement;
                var c = new Ctx();
                c.Day = r.GetProperty("day").GetString();
                c.Start = r.GetProperty("start").GetInt32();
                c.N = r.GetProperty("n").GetInt32();
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
                    var p = new PriorDay();
                    p.High = pd.GetProperty("high").GetDouble();
                    p.Low = pd.GetProperty("low").GetDouble();
                    p.Close = pd.GetProperty("close").GetDouble();
                    c.Prior.Add(p);
                }
                return c;
            }
        }
        static double[] ReadD(JsonElement r, string k)
        {
            return r.GetProperty(k).EnumerateArray().Select(x => x.GetDouble()).ToArray();
        }

        // Byte-format-identical to research/nt8_port/csharp/Program.WriteDay.
        static void WriteDay(string path, string day, List<BarRec> bars)
        {
            var sb = new StringBuilder();
            sb.Append("{\"day\":\"").Append(day).Append("\",\"bars\":[");
            for (int b = 0; b < bars.Count; b++)
            {
                var bar = bars[b];
                if (b > 0) sb.Append(',');
                sb.Append("{\"bar_ts\":").Append(bar.BarTs);
                foreach (var d in ModelData.Topk) sb.Append(",\"f_").Append(d).Append("\":").Append(bar.F[d]);
                sb.Append(",\"gov_stream\":\"").Append(bar.Gov).Append('"');
                sb.Append(",\"gov_dir\":").Append(bar.GovDir);
                sb.Append(",\"P_compact\":").Append(Pd.Fin(bar.P) ? bar.P.ToString("R", CultureInfo.InvariantCulture) : "null");
                sb.Append(",\"entry\":").Append(bar.Entry);
                sb.Append(",\"entry_dir\":").Append(bar.EntryDir);
                sb.Append(",\"zz_leg\":").Append(bar.ZzLeg);
                sb.Append(",\"zz_confirm\":").Append(bar.ZzConfirm);
                sb.Append(",\"zz_pivot_age_min\":").Append(bar.ZzPivAge.ToString("R", CultureInfo.InvariantCulture));
                sb.Append(",\"zz_pivot_price\":").Append(bar.ZzPivPrice.ToString("R", CultureInfo.InvariantCulture));
                sb.Append('}');
            }
            sb.Append("]}");
            File.WriteAllText(path, sb.ToString());
        }
    }
}
