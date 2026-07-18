// Gens.cs -- the 22 top-K stream generators (minus TMPL0, in Tmpl0.cs), ported verbatim
// from research/nt8_catalog/tools/dossier_signal_pipeline.py. Each returns List<Fire>.
using System;
using System.Collections.Generic;

namespace Nt8Port
{
    static class Gens
    {
        const int COOLDOWN = 60;

        // ---- ZIGZAG -----------------------------------------------------------------
        public static List<Fire> Zigzag(Ctx x)
        {
            var o = new List<Fire>();
            foreach (int i in x.RthIdx())
                if (x.PivConfirm[i] != 0)
                    o.Add(x.Emit(i, x.PivConfirm[i] > 0, Pd.Fin(x.ZzThr[i]) ? x.ZzThr[i] : 0.0, "ZIGZAG"));
            return o;
        }

        // ---- ORB-02 -----------------------------------------------------------------
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

        // ---- ROUND-05 ---------------------------------------------------------------
        public static List<Fire> Round05(Ctx x)
        {
            const double GRID = 50.0, PRIME = 5.0;
            var o = new List<Fire>();
            var primB = new Dictionary<double, bool>(); var primS = new Dictionary<double, bool>();
            for (int i = 0; i < x.N; i++)
            {
                double p = x.C[i];
                double base_ = (double)((long)(p / GRID)) * GRID;  // int() trunc toward zero
                double[] levels = { base_ - GRID, base_, base_ + GRID };
                foreach (double L in levels)
                {
                    if (p >= L && primB.TryGetValue(L, out bool bb) && bb)
                    { primB[L] = false; if (x.Rth[i] && i >= x.Start) o.Add(x.Emit(i, true, PRIME, "ROUND05")); }
                    if (p <= L && primS.TryGetValue(L, out bool ss) && ss)
                    { primS[L] = false; if (x.Rth[i] && i >= x.Start) o.Add(x.Emit(i, false, PRIME, "ROUND05")); }
                    if (p < L - PRIME) primB[L] = true; else if (p >= L) primB[L] = false;
                    if (p > L + PRIME) primS[L] = true; else if (p <= L) primS[L] = false;
                }
            }
            return o;
        }

        // ---- VWAP-03 ----------------------------------------------------------------
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
                if (buf.Count < 20) continue;   // zprev unchanged (matches python)
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

        // ---- DOW-19 -----------------------------------------------------------------
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

        // ---- TUNNEL-20 --------------------------------------------------------------
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

        // ---- ATR-09 -----------------------------------------------------------------
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

        // ---- PIVOT-16 ---------------------------------------------------------------
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

        // ---- RENKO-24 ---------------------------------------------------------------
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

        // ---- SAR-23 -----------------------------------------------------------------
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

        // ---- RSI-06 -----------------------------------------------------------------
        public static List<Fire> Rsi06(Ctx x)
        {
            var delta = Pd.Diff(x.C);
            var up = new double[x.N]; var dn = new double[x.N];
            for (int i = 0; i < x.N; i++)
            {
                if (!Pd.Fin(delta[i])) { up[i] = double.NaN; dn[i] = double.NaN; continue; }
                up[i] = Math.Max(delta[i], 0.0);     // delta.clip(lower=0)
                dn[i] = -Math.Min(delta[i], 0.0);    // -delta.clip(upper=0)
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

        // ---- MACD-07 ----------------------------------------------------------------
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

        // ---- CTX-ER -----------------------------------------------------------------
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

        // ---- EXIT-KMDR --------------------------------------------------------------
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

        // ---- PTRN-ENGULF ------------------------------------------------------------
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

        // ---- NMP helpers ------------------------------------------------------------
        const double Z_ENTRY = 1.8481, Z_EXIT = 0.4752, NMP_EPS = 0.1; const int NMP_K = 21;

        // per-row vr from 1m buckets: vr_by_bucket[last-closed]; NaN if bucket absent
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
                outv[i] = b.Pos.TryGetValue(want, out int pp) ? vrb[pp] : double.NaN;
            }
            return outv;
        }

        // (row, z) episode fires
        static List<(int i, double z)> NmpFires(Ctx x)
        {
            var o = new List<(int, double)>(); bool armed = true;
            for (int i = 0; i < x.N; i++)
            {
                double zi = x.Zse[i]; if (!Pd.Fin(zi)) continue;
                if (Math.Abs(zi) < Z_EXIT) armed = true;
                else if (armed && Math.Abs(zi) > Z_ENTRY) { armed = false; if (x.Rth[i] && i >= x.Start) o.Add((i, zi)); }
            }
            return o;
        }

        // per-row lambda_hat (ffill), needs zse
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
            // ffill
            double last = double.NaN;
            for (int i = 0; i < x.N; i++) { if (Pd.Fin(lam[i])) last = lam[i]; else lam[i] = last; }
            return lam;
        }

        // ---- NMP --------------------------------------------------------------------
        public static List<Fire> Nmp(Ctx x)
        {
            if (!x.HasZse) return new List<Fire>();
            var vr = Vr1m(x); var o = new List<Fire>();
            foreach (var (i, zi) in NmpFires(x))
                if (Pd.Fin(vr[i]) && vr[i] < 1.0) o.Add(x.Emit(i, zi < 0, Math.Abs(zi), "NMP"));
            return o;
        }

        // ---- NMP9 waterfall (verbatim original 2026-04-08) --------------------------
        // Returns events (row, isLong, tier, value). Tier is dashless (e.g. RIDEAGAINST).
        static List<(int i, bool isLong, string tier, double val)> Nmp9Events(Ctx x)
        {
            var ev = new List<(int, bool, string, double)>();
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
                bool longDir = z <= 0;                     // 'short' if z>0 else 'long'
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
                        bool rdirLong = !longDir;          // flip the fade
                        if (absVel >= VELT) { tier = "RIDEMOM"; rl = rdirLong; val = absVel; }
                        else { tier = "RIDECALM"; rl = rdirLong; val = Math.Abs(z); }
                    }
                    else if (absVel >= VELT) { tier = "FADEMOM"; rl = longDir; val = absVel; }
                    else { tier = "FADECALM"; rl = longDir; val = Math.Abs(z); }
                }
                string key = tier == null ? null : (rl ? "long" : "short") + "|" + tier;
                if (tier != null && key != prev) ev.Add((i, rl, "NMP9" + tier, val));
                prev = key;
            }
            return ev;
        }

        public static List<Fire> Nmp9(Ctx x, string tier)
        {
            var o = new List<Fire>();
            foreach (var (i, isLong, t, val) in Nmp9Events(x))
                if (t == "NMP9" + tier) o.Add(x.Emit(i, isLong, val, "NMP9" + tier));
            return o;
        }

        // ---- NMPT waterfall (2026-04-18 re-derivation) ------------------------------
        static List<(int i, bool isLong, string tier, double val)> NmptEvents(Ctx x)
        {
            var ev = new List<(int, bool, string, double)>();
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
                    bool bdirLong = z > 0;                 // 'long' if z>0 else 'short'
                    if ((bdirLong && dmi > -5) || (!bdirLong && dmi < 5)) { tier = "MTFBRK"; rl = bdirLong; val = Math.Min(z5, z15); }
                }
                else
                {
                    bool hiOpp = (longDir && v5vel < -3 && h1vel < -3) || (!longDir && v5vel > 3 && h1vel > 3);
                    if (!hiOpp) { tier = "FADECALM"; rl = longDir; val = Math.Abs(z); }
                }
                string key = tier == null ? null : (rl ? "long" : "short") + "|" + tier;
                if (tier != null && key != prev) ev.Add((i, rl, "NMPT" + tier, val));
                prev = key;
            }
            return ev;
        }

        public static List<Fire> Nmpt(Ctx x, string tier)
        {
            var o = new List<Fire>();
            foreach (var (i, isLong, t, val) in NmptEvents(x))
                if (t == "NMPT" + tier) o.Add(x.Emit(i, isLong, val, "NMPT" + tier));
            return o;
        }

        // ---- shared -----------------------------------------------------------------
        static int BktRow(Ctx x, Buckets b, int k)
        {
            int r = b.CloseRow[k];
            if (r < 0 || r < x.Start || !x.Rth[r]) return -1;
            return r;
        }
        static double[] NullArr(int n) { var a = new double[n]; for (int i = 0; i < n; i++) a[i] = double.NaN; return a; }
    }
}
