"""Every question worked through in the 2026-05-22 trade-outcome session,
as one function each. Each returns (title, verdict, markdown_body).

All units are dollars (MNQ = $2/point; pnl_usd is net of $6/leg friction).
IS and OOS are reported separately, never pooled.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from excursions import (pct, pct1, usd, md_table, bootstrap_ci,  # noqa: E402
                        FRICTION_USD, MIN_CELL_N)

STEP = 20  # dollars per drawdown iteration


# --- Q1 -------------------------------------------------------------------
def q01_distributions(IS, OOS):
    title = 'Q1 - Distributions of entry-to-close, MAE, MFE'
    hdr = ['level $', 'MAE≥ IS', 'MFE≥ IS', 'close≥+ IS',
           'close≤- IS', 'MAE≥ OOS', 'MFE≥ OOS',
           'close≥+ OOS', 'close≤- OOS']
    rows = []
    for L in range(0, 401, 25):
        r = [f'${L}']
        for d in (IS, OOS):
            r += [pct((d['mae_usd'] >= L).mean()), pct((d['mfe_usd'] >= L).mean()),
                  pct((d['close_usd'] >= L).mean()), pct((d['close_usd'] <= -L).mean())]
        rows.append(r)
    body = (f'## {title}\n\n'
            'Survival (exceedance) probability that each per-trade quantity '
            'reaches a given dollar magnitude. The foundation table.\n\n'
            + md_table(hdr, rows))
    verdict = (f'Median MFE ${IS["mfe_usd"].median():.0f}/${OOS["mfe_usd"].median():.0f} '
               f'(IS/OOS); P(close>0) {pct((IS["close_usd"]>0).mean())}/'
               f'{pct((OOS["close_usd"]>0).mean())}.')
    return title, verdict, body


# --- Q2 -------------------------------------------------------------------
def q02_joint(IS, OOS):
    title = 'Q2 - Joint MFE x MAE -> P(close>0)'
    mfe_e = [0, 50, 100, 150, 200, 300, np.inf]
    mae_e = [0, 25, 50, 100, 200, np.inf]
    mlbl = ['0-50', '50-100', '100-150', '150-200', '200-300', '300+']
    albl = ['0-25', '25-50', '50-100', '100-200', '200+']
    out = []
    for name, d in (('IS', IS), ('OOS', OOS)):
        mb = pd.cut(d['mfe_usd'], mfe_e, right=False, labels=mlbl)
        ab = pd.cut(d['mae_usd'], mae_e, right=False, labels=albl)
        g = d.assign(_m=mb, _a=ab).groupby(['_m', '_a'], observed=False)['pnl_usd']
        pw = g.apply(lambda s: (s > 0).mean()).unstack()
        n = g.size().unstack()
        rows = []
        for m in mlbl:
            r = [m]
            for a in albl:
                cnt = n.loc[m, a]
                r.append('-' if cnt == 0 else f'{pw.loc[m, a]*100:.0f}% / {int(cnt)}')
            rows.append(r)
        out.append(f'**{name}** (cell = P(close>0) / n)\n\n'
                   + md_table(['MFE \\ MAE'] + albl, rows))
    body = (f'## {title}\n\nDoes the drawdown a trade suffered (MAE) change its '
            'win odds, once you know its peak (MFE)?\n\n' + '\n\n'.join(out))
    return title, 'P(close>0) is driven almost entirely by MFE; MAE barely moves it until the extreme corner.', body


# --- Q3 -------------------------------------------------------------------
def _cont_rows(d, targets):
    rows = []
    for x in range(0, 401, 25):
        m = d['mfe_usd'] >= x
        n = int(m.sum())
        if n == 0:
            continue
        p = d.loc[m, 'pnl_usd'].values
        lo, hi = bootstrap_ci((p > 0).astype(float))
        flag = ' !' if n < MIN_CELL_N else ''
        rows.append([f'${x}{flag}', f'{n:,}']
                    + [pct((p >= t).mean()) for t in targets]
                    + [pct((p > 0).mean()), usd(np.mean(p)),
                       f'[{pct(lo)}, {pct(hi)}]'])
    return rows


def q03_continuation(IS, OOS):
    title = 'Q3 - Continuation: given up +$x, where does it close?'
    targets = [50, 100, 200]
    hdr = ['up +$x', 'n'] + [f'close≥${t}' for t in targets] + \
          ['close>0', 'mean close', 'P(close>0) 95% CI']
    body = [f'## {title}', '',
            'Condition: the trade has reached open profit +$x (MFE≥x). '
            'Distribution of the FINAL close.', '',
            '**IS**', '', md_table(hdr, _cont_rows(IS, targets)), '',
            '**OOS**', '', md_table(hdr, _cont_rows(OOS, targets))]
    return title, 'A big close needs a big peak; "currently green" alone says little beyond how high.', '\n'.join(body)


# --- Q4 -------------------------------------------------------------------
def q04_reach_conditional(IS, OOS):
    title = 'Q4 - Conditional: at +$x, P(it continues another step)'
    bump = 50
    hdr = ['at +$x', 'n IS', 'P(peak≥x+$50) IS', 'P(close≥x) IS',
           'n OOS', 'P(peak≥x+$50) OOS', 'P(close≥x) OOS']
    rows = []
    for x in range(50, 351, 50):
        r = [f'${x}']
        for d in (IS, OOS):
            m = d['mfe_usd'] >= x
            n = int(m.sum())
            if n == 0:
                r += ['0', 'n/a', 'n/a']
                continue
            ext = (d.loc[m, 'mfe_usd'] >= x + bump).mean()
            hold = (d.loc[m, 'close_usd'] >= x).mean()
            r += [f'{n:,}', pct(ext), pct(hold)]
        rows.append(r)
    body = (f'## {title}\n\nThe "$100 -> $150" question, generalised. Given the '
            'trade is up +$x: does the peak push another $50, and does it hold '
            'the +$x to the close?\n\n' + md_table(hdr, rows))
    return title, 'Peak extends ~50% at every level; "holds the +$x" is only ~40-50% - the structural ~1R giveback.', body


# --- Q5 -------------------------------------------------------------------
def q05_cut_vs_hold_winner(IS, OOS):
    title = 'Q5 - Cut-and-bank a winner vs hold to the exit'
    hdr = ['at +$L', 'n IS', 'HOLD mean IS', 'hold-cut IS',
           'n OOS', 'HOLD mean OOS', 'hold-cut OOS']
    rows = []
    for L in [100, 150, 200, 250, 300, 400, 500]:
        r = [f'${L}']
        for d in (IS, OOS):
            m = d['mfe_usd'] >= L
            n = int(m.sum())
            if n < 5:
                r += ['-', '-', '-']
                continue
            hold = float(d.loc[m, 'pnl_usd'].mean())
            r += [f'{n:,}', usd(hold), usd(hold - (L - FRICTION_USD))]
        rows.append(r)
    body = (f'## {title}\n\n"Cut" banks +$L minus ${FRICTION_USD:.0f} friction. '
            '"Hold" runs to the R-trigger exit. `hold-cut` > 0 means holding wins.'
            '\n\n' + md_table(hdr, rows))
    return title, 'Holding beats cutting at every level, IS and OOS - the right tail pays for the giveback.', body


# --- Q6 -------------------------------------------------------------------
def q06_giveback(IS, OOS):
    title = 'Q6 - Giveback: how much of the peak survives to the close'
    buckets = [(50, 100), (100, 150), (150, 200), (200, 300), (300, 1e9)]
    hdr = ['MFE peak', 'n IS', 'median close IS', 'capture% IS', 'gave-back≤$20 IS',
           'n OOS', 'median close OOS', 'capture% OOS']
    rows = []
    for lo, hi in buckets:
        lbl = f'${lo}-{hi}' if hi < 1e9 else f'${lo}+'
        r = [lbl]
        for d in (IS, OOS):
            b = d[(d['mfe_usd'] >= lo) & (d['mfe_usd'] < hi)]
            if len(b) == 0:
                r += ['0', '-', '-'] + (['-'] if d is IS else [])
                continue
            cap = (b['pnl_usd'] / b['mfe_usd']).median()
            cells = [f'{len(b):,}', usd(b['pnl_usd'].median()), pct(cap)]
            if d is IS:
                cells.append(pct((b['pnl_usd'] <= 20).mean()))
            r += cells
        rows.append(r)
    body = (f'## {title}\n\nThe peak->close drop averages '
            f'${(IS["mfe_usd"]-IS["pnl_usd"]).mean():.0f} IS / '
            f'${(OOS["mfe_usd"]-OOS["pnl_usd"]).mean():.0f} OOS -- a roughly '
            'FIXED ~1R toll, so it eats small excursions whole and barely '
            'dents big ones.\n\n' + md_table(hdr, rows))
    return title, 'Giveback is a fixed ~1R toll: devastating on $50-100 peaks, minor on $300+.', body


# --- Q7 -------------------------------------------------------------------
def q07_mfe_close_cumulative(IS, OOS):
    title = 'Q7 - Given an MFE of +$300, where does it close (cumulative)'
    D = 300
    bands = [(300, 1e9, '≥ +$300 (ran further)'), (250, 300, '+$250-300'),
             (200, 250, '+$200-250'), (150, 200, '+$150-200'),
             (100, 150, '+$100-150'), (50, 100, '+$50-100'),
             (0, 50, '+$0-50'), (-1e9, 0, '< $0 (gave it all back)')]
    out = []
    for name, d in (('IS', IS), ('OOS', OOS)):
        c = d.loc[d['mfe_usd'] >= D, 'pnl_usd'].values
        rows, cum = [], 100.0
        for lo, hi, lbl in bands:
            p = ((c >= lo) & (c < hi)).mean() * 100.0
            rows.append([lbl, f'{p:.1f}%', f'{cum:.1f}%'])
            cum -= p
        out.append(f'**{name}** (n={len(c)})\n\n'
                   + md_table(['close lands in', 'P(bucket)', 'cum P(close≤top)'], rows))
    body = (f'## {title}\n\nEach bucket is distinct and sums to 100%; the '
            'cumulative column is P(close at or below the band top).\n\n'
            + '\n\n'.join(out))
    return title, 'From a $300 peak the close clusters in $200-300; below $0 is ~1% - it does not "give it all back".', body


# --- Q8 -------------------------------------------------------------------
def q08_equity_loss_map(IS, OOS):
    title = 'Q8 - Equity-loss map: P(close<0) by MFE reached'
    hdr = ['MFE reached ≥', 'n IS', 'P(lose equity) IS', 'P(close<-$100) IS',
           'n OOS', 'P(lose equity) OOS', 'P(close<-$100) OOS']
    rows = []
    for L in [25, 50, 75, 100, 150, 200, 250, 300]:
        r = [f'+${L}']
        for d in (IS, OOS):
            b = d[d['mfe_usd'] >= L]
            r += [f'{len(b):,}', pct1((b['pnl_usd'] < 0).mean()),
                  pct1((b['pnl_usd'] < -100).mean())]
        rows.append(r)
    body = (f'## {title}\n\n"Lose equity" = closes red. The question behind the '
            '"safety limit" idea: how far must a trade run before it is safe '
            'from a negative close?\n\n' + md_table(hdr, rows))
    return title, 'Equity loss is a low-MFE phenomenon - by +$150 MFE it is ~1%; a "$250 safety limit" guards empty space.', body


# --- Q9 -------------------------------------------------------------------
def _rec_rows(d):
    rows = []
    for D in range(0, 261, 20):
        m = d['mae_usd'] >= D
        n = int(m.sum())
        if n < 10:
            break
        p = d.loc[m, 'pnl_usd'].values
        lo, hi = bootstrap_ci((p > 0).astype(float))
        flag = ' !' if n < MIN_CELL_N else ''
        rows.append([f'-${D}{flag}', f'{n:,}', pct((p > 0).mean()),
                     pct((p >= 100).mean()), usd(np.mean(p)),
                     f'[{pct(lo)}, {pct(hi)}]'])
    return rows


def q09_recovery(IS, OOS):
    title = 'Q9 - Recovery: given down -$d, does it work out?'
    hdr = ['down -$d', 'n', 'P(close>0)', 'P(close≥+$100)', 'mean close',
           'P(close>0) 95% CI']
    body = (f'## {title}\n\nCondition: the trade has drawn down to -$d '
            '(MAE≥d). Distribution of the final close.\n\n'
            '**IS**\n\n' + md_table(hdr, _rec_rows(IS)) + '\n\n'
            '**OOS**\n\n' + md_table(hdr, _rec_rows(OOS)))
    return title, 'Recovery to green is ~18-28% and erodes only gently with depth; mean close negative for every d>=$40.', body


# --- Q10 ------------------------------------------------------------------
def q10_mae_recovery_sweep(IS, OOS):
    title = 'Q10 - Full MAE -> close sweep (every $20)'
    hdr = ['down -$D', 'n', 'P(≥0) recover', 'P(≥+$50)', 'P(≥+$100)',
           'mean close', 'median close', 'hold-bail']
    out = []
    for name, d in (('IS', IS), ('OOS', OOS)):
        rows, D = [], 0
        while True:
            b = d[d['mae_usd'] >= D]
            if len(b) < 10:
                break
            c = b['pnl_usd'].values
            flag = ' !' if len(b) < MIN_CELL_N else ''
            rows.append([f'-${D}{flag}', f'{len(b):,}', pct((c >= 0).mean()),
                         pct((c >= 50).mean()), pct((c >= 100).mean()),
                         usd(np.mean(c)), usd(np.median(c)), usd(np.mean(c) + D)])
            D += STEP
        out.append(f'**{name}**\n\n' + md_table(hdr, rows))
    body = (f'## {title}\n\nThe complete recovery table, fine-grained. '
            '`hold-bail` = mean close minus the -$D you would lock by bailing now.'
            '\n\n' + '\n\n'.join(out))
    return title, 'hold-bail is positive at every depth - bailing always locks the worst version.', body


# --- Q11 ------------------------------------------------------------------
def q11_mae_worsening(IS, OOS):
    title = 'Q11 - Probability a drawdown gets WORSE'
    hdr = ['at -$D', 'n', 'P(deepen +$20)', 'P(close<-$D worse)',
           'P(stuck -D..0)', 'P(recover ≥0)']
    out = []
    for name, d in (('IS', IS), ('OOS', OOS)):
        rows, D = [], 0
        while True:
            b = d[d['mae_usd'] >= D]
            n = len(b)
            if n < 10:
                break
            c = b['pnl_usd'].values
            deeper = (d['mae_usd'] >= D + STEP).sum() / n
            flag = ' !' if n < MIN_CELL_N else ''
            rows.append([f'-${D}{flag}', f'{n:,}', pct(deeper),
                         pct((c < -D).mean()),
                         pct(((c >= -D) & (c < 0)).mean()), pct((c >= 0).mean())])
            D += STEP
        out.append(f'**{name}**\n\n' + md_table(hdr, rows))
    body = (f'## {title}\n\n`P(deepen)` = given at -$D, probability the drawdown '
            'extends another $20. NOTE: this column rises in IS but falls in OOS '
            '- the "drawdowns gain momentum" effect does NOT replicate.\n\n'
            + '\n\n'.join(out))
    return title, 'P(deepen) rises in IS, falls in OOS - regime-dependent, not bankable. Recovery odds erode gently in both.', body


# --- Q12 ------------------------------------------------------------------
def q12_mae_iteration(IS, OOS):
    title = 'Q12 - Iterative drawdown chain (n -> n+1)'
    hdr = ['n', 'drawdown', 'p_reach(n) IS', 'p_advance IS', 'p_recover IS',
           'p_reach(n) OOS', 'p_advance OOS', 'p_recover OOS']
    rows = []
    n = 0
    prev = {'IS': len(IS), 'OOS': len(OOS)}
    while True:
        level = n * STEP
        cells, alive = [str(n), f'-${level}'], False
        for name, d in (('IS', IS), ('OOS', OOS)):
            here = int((d['mae_usd'] >= level).sum())
            if here >= 10:
                alive = True
            c = d.loc[d['mae_usd'] >= level, 'pnl_usd'].values
            adv = here / prev[name] * 100 if prev[name] else 0
            cells += [pct1(here / len(d)),
                      f'{adv:.0f}%' if here else '-',
                      pct((c >= 0).mean()) if len(c) else '-']
            prev[name] = here if here else prev[name]
        if not alive:
            break
        rows.append(cells)
        n += 1
    body = (f'## {title}\n\nEach iteration n deepens the drawdown by ${STEP}. '
            '`p_advance` = P(reach step n | reached n-1); `p_reach` = cumulative '
            'from entry; `p_recover` = P(close≥0 | here).\n\n'
            + md_table(hdr, rows))
    return title, 'p_reach decays steeply (deep drawdowns are rare); p_advance is regime-split; p_recover erodes gently.', body


# --- Q13 ------------------------------------------------------------------
def q13_cut_vs_hold_loser(IS, OOS):
    title = 'Q13 - Cut a loser vs hold to the exit'
    hdr = ['at -$D', 'n IS', 'HOLD mean IS', 'hold-bail IS',
           'n OOS', 'HOLD mean OOS', 'hold-bail OOS']
    rows = []
    for D in [40, 60, 80, 100, 140]:
        r = [f'-${D}']
        for d in (IS, OOS):
            m = d['mae_usd'] >= D
            n = int(m.sum())
            if n < 5:
                r += ['-', '-', '-']
                continue
            hold = float(d.loc[m, 'pnl_usd'].mean())
            r += [f'{n:,}', usd(hold), usd(hold - (-D))]
        rows.append(r)
    body = (f'## {title}\n\n"Cut" locks -$D. "Hold" runs to the R-trigger exit. '
            '`hold-bail` > 0 means cutting loses money.\n\n' + md_table(hdr, rows))
    return title, 'Cutting loses at every drawdown level - the R-trigger recovers ~1R off the low; bailing forfeits it.', body


# --- Q14 ------------------------------------------------------------------
def q14_timing(IS, OOS):
    title = 'Q14 - When does the MAE happen, and how long do recoverers take?'
    hdr = ['drew down ≥', 'n', 't->bottom (min)', 'bottom @ %dur',
           'RECOVER dur (min)', 'NO-REC dur (min)']
    out = []
    for name, d in (('IS', IS), ('OOS', OOS)):
        rows = []
        for D in [20, 40, 60, 80, 100]:
            b = d[d['mae_usd'] >= D]
            if len(b) < 10:
                continue
            rec = b[b['pnl_usd'] >= 0]
            nor = b[b['pnl_usd'] < 0]
            rows.append([f'-${D}', f'{len(b):,}',
                         f'{b["t_to_bottom_min"].median():.1f}',
                         pct(b['frac_to_bottom'].median()),
                         f'{rec["dur_min"].median():.1f}' if len(rec) else '-',
                         f'{nor["dur_min"].median():.1f}' if len(nor) else '-'])
        out.append(f'**{name}**\n\n' + md_table(hdr, rows))
    body = (f'## {title}\n\nThe worst point lands ~3-4 min in (constant with '
            'depth) but ~83-100% of the way through the trade. Recoverers run '
            'roughly 2x as long as non-recoverers.\n\n' + '\n\n'.join(out))
    return title, 'Recoverers run ~2x longer than non-recoverers - trade AGE is informative where depth was not.', body


# --- Q15 ------------------------------------------------------------------
def q15_bimodal(IS, OOS):
    title = 'Q15 - The bimodal split: winners vs losers, not "peak then collapse"'
    hdr = ['MAE-bottom mode', 'n', 'median MFE', 'MFE≥$50',
           'peak-before-trough', 'median close', 'loss rate']

    def mk(b):
        if len(b) == 0:
            return ['n=0', '', '', '', '', '']
        return [f'{len(b):,}', usd(b['mfe_usd'].median()),
                pct((b['mfe_usd'] >= 50).mean()),
                pct((b['mfe_bar'] < b['mae_bar']).mean()),
                usd(b['pnl_usd'].median()), pct((b['pnl_usd'] < 0).mean())]
    out = []
    for name, d in (('IS', IS), ('OOS', OOS)):
        dd = d[d['mae_usd'] >= 40]
        rows = [['EARLY bottom (q1)'] + mk(dd[dd['frac_to_bottom'] < 0.25]),
                ['LATE bottom (q4)'] + mk(dd[dd['frac_to_bottom'] >= 0.75]),
                ['all losers'] + mk(d[d['pnl_usd'] < 0])]
        out.append(f'**{name}** (drew down ≥$40)\n\n' + md_table(hdr, rows))
    body = (f'## {title}\n\nTesting the claim "bimodal = peaks MFE then goes '
            'negative". The ordering holds, but the "peak" in the losing mode '
            'is ~$14 - noise, not a peak.\n\n' + '\n\n'.join(out))
    return title, 'Not "peak then collapse" - it is winners (real $100+ MFE) vs losers (~$14 poke, never worked).', body


QUESTIONS = [
    q01_distributions, q02_joint, q03_continuation, q04_reach_conditional,
    q05_cut_vs_hold_winner, q06_giveback, q07_mfe_close_cumulative,
    q08_equity_loss_map, q09_recovery, q10_mae_recovery_sweep,
    q11_mae_worsening, q12_mae_iteration, q13_cut_vs_hold_loser,
    q14_timing, q15_bimodal,
]
