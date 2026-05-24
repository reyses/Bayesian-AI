"""v9c — combined CLASSIFICATION × MFE_REGRESSION fire score.

Per user 2026-05-12: "(p) needs to filter also for PnL — trade needs ≥$10".

For each bar:
    p_entry_short    = GBM classifier(state)
    mfe_short_pred   = GBM regressor(state)
    score_short      = p_entry_short × mfe_short_pred

Fire SHORT when:
    score_short >= MIN_SCORE  (e.g. 8 = 0.55 prob × $15 expected MFE)
    AND mfe_short_pred >= MIN_MFE_DOLLARS  (viability)
    AND score_short >= score_long  (winning direction)

This combines:
  - HIGH probability (model thinks it's an oracle entry)
  - HIGH expected $ MFE (worth firing — passes viability)
  - DIRECTIONAL EDGE (this side bigger than opposite)
"""
from __future__ import annotations
import argparse, csv, pickle, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.cusp_marker import compute_anchor
from tools.sim_decay_rules import load_1m_bars
from tools.regret_1m_oracle import extract_state_vector, TICK_DOLLAR


TICK = 0.25
OUT_DIR = Path('reports/findings/sim_v9c')
DECISION_MODELS_PATH = Path('reports/findings/logistic_oracles/decision_models.pkl')
MFE_MODELS_PATH = Path('reports/findings/logistic_oracles/mfe_regressors.pkl')


def _ts(d): return datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def simulate(t_start, t_end, dec_models, mfe_models,
                p_thr: float = 0.50, mfe_thr: float = 15.0,
                score_thr: float = 8.0,
                hard_stop_ticks: float = 30, max_hold_min: int = 60,
                cooldown_min: int = 0):
    df = load_1m_bars(t_start, t_end)
    if df.empty: return []
    ts    = df['timestamp'].values.astype(np.int64)
    close = df['close'].values.astype(float)
    high  = df['high'].values.astype(float)
    low   = df['low'].values.astype(float)
    print(f'  {len(df)} bars; computing anchors...')
    M_15s, S_15s = compute_anchor('15s', ts, t_start, t_end, window=20, column='close')
    M_1m,  S_1m  = compute_anchor('1m',  ts, t_start, t_end, window=15, column='close')
    M_15m, S_15m = compute_anchor('15m', ts, t_start, t_end, window=12, column='close')
    Mh,    Sh    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='high')
    Ml,    Sl    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='low')
    Mc,    Sc    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='close')

    es = dec_models['entry_short']; el = dec_models['entry_long']
    xs = dec_models['exit_short'];  xl = dec_models['exit_long']
    feature_cols = es['feature_cols']

    print(f'  extracting decision features for {len(ts)} bars...')
    X = np.zeros((len(ts), len(feature_cols)), dtype=float)
    for i in range(15, len(ts)):
        state = extract_state_vector(i, close, M_15s, S_15s, M_1m, S_1m,
                                                    M_15m, S_15m, Mh, Sh, Ml, Sl, Mc, Sc)
        for j, fc in enumerate(feature_cols):
            v = state.get(fc, 0.0)
            X[i, j] = 0.0 if v is None else float(v)

    print('  batch-predicting classifications + MFE...')
    p_es = es['gbm_model'].predict_proba(X)[:, 1]
    p_el = el['gbm_model'].predict_proba(X)[:, 1]
    p_xs = xs['gbm_model'].predict_proba(X)[:, 1]
    p_xl = xl['gbm_model'].predict_proba(X)[:, 1]
    mfe_short_pred = mfe_models['mfe_short_dollars']['reg_model'].predict(X)
    mfe_long_pred  = mfe_models['mfe_long_dollars']['reg_model'].predict(X)

    # Score = P × predicted_MFE
    score_short = p_es * mfe_short_pred
    score_long  = p_el * mfe_long_pred

    print('  walking bar-by-bar...')
    trades = []
    open_t = None
    cooldown_until = 0.0

    for i in range(15, len(ts)):
        if open_t is not None:
            if open_t['side'] == 'LONG':
                pnl_ticks = (close[i] - open_t['entry_price']) / TICK
            else:
                pnl_ticks = (open_t['entry_price'] - close[i]) / TICK
            open_t['peak'] = max(open_t['peak'], pnl_ticks)
            open_t['worst'] = min(open_t['worst'], pnl_ticks)
            dur = (ts[i] - open_t['entry_ts']) / 60.0
            close_now = False; reason = ''
            if pnl_ticks <= -hard_stop_ticks:
                close_now = True; reason = 'hard_stop'
            elif open_t['side']=='SHORT' and p_xs[i] >= 0.55:
                close_now = True; reason = f'p_exit_short={p_xs[i]:.2f}'
            elif open_t['side']=='LONG' and p_xl[i] >= 0.55:
                close_now = True; reason = f'p_exit_long={p_xl[i]:.2f}'
            elif dur >= max_hold_min:
                close_now = True; reason = 'time_stop'
            if close_now:
                open_t['exit_ts'] = ts[i]; open_t['exit_price'] = close[i]
                open_t['reason'] = reason; open_t['dur_min'] = dur
                trades.append(open_t); open_t = None
                cooldown_until = ts[i] + cooldown_min * 60
            else:
                continue
        if ts[i] < cooldown_until: continue

        # ── COMBINED gate: P × MFE_predicted ──
        short_pass = (p_es[i] >= p_thr) and (mfe_short_pred[i] >= mfe_thr) and (score_short[i] >= score_thr)
        long_pass  = (p_el[i] >= p_thr) and (mfe_long_pred[i]  >= mfe_thr) and (score_long[i]  >= score_thr)

        if short_pass and score_short[i] >= score_long[i]:
            open_t = {'side':'SHORT', 'entry_ts':int(ts[i]), 'entry_price':float(close[i]),
                          'p_at_entry':float(p_es[i]),
                          'mfe_pred':float(mfe_short_pred[i]),
                          'score':float(score_short[i]),
                          'peak':0.0, 'worst':0.0}
        elif long_pass:
            open_t = {'side':'LONG', 'entry_ts':int(ts[i]), 'entry_price':float(close[i]),
                          'p_at_entry':float(p_el[i]),
                          'mfe_pred':float(mfe_long_pred[i]),
                          'score':float(score_long[i]),
                          'peak':0.0, 'worst':0.0}

    if open_t is not None:
        open_t['exit_ts'] = int(ts[-1]); open_t['exit_price'] = float(close[-1])
        open_t['reason'] = 'eod'; open_t['dur_min'] = (ts[-1] - open_t['entry_ts']) / 60.0
        trades.append(open_t)
    return trades


def report_stats(trades):
    if not trades: return None
    pnls = np.array([(t['exit_price']-t['entry_price'])/TICK*TICK_DOLLAR if t['side']=='LONG'
                          else (t['entry_price']-t['exit_price'])/TICK*TICK_DOLLAR
                          for t in trades])
    n = len(trades)
    n_win = (pnls > 0).sum()
    win = pnls[pnls > 0].sum() if any(pnls > 0) else 0
    lose = -pnls[pnls < 0].sum() if any(pnls < 0) else 0
    pf = (win/lose - 1) if lose > 0 else float('inf')
    from collections import defaultdict
    daily = defaultdict(float)
    for t, p in zip(trades, pnls):
        d = datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).strftime('%Y-%m-%d')
        daily[d] += p
    nd = max(1, len(daily))
    return {
        'n': n, 'win_rate': 100*n_win/n,
        'pf_wr': pf, 'total': pnls.sum(), 'per_day': pnls.sum()/nd,
        'mean_per_trade': pnls.mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--name', default='OOS')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--p-thr', type=float, default=0.40)
    ap.add_argument('--mfe-thr', type=float, default=0.0)
    ap.add_argument('--score-thr', type=float, default=0.0)
    args = ap.parse_args()

    with open(DECISION_MODELS_PATH, 'rb') as f:
        dec_models_list = pickle.load(f)
    dec_models = {m['name']: m for m in dec_models_list}
    with open(MFE_MODELS_PATH, 'rb') as f:
        mfe_models = pickle.load(f)
    print('Loaded models.')

    t_start = _ts(args.start); t_end = _ts(args.end) + 86400

    if args.sweep:
        print(f'\n=== SWEEP ({args.name}) ===')
        print(f'{"P_thr":>6} {"MFE_thr":>8} {"score_thr":>10} {"n":>6} {"wr":>6} {"PF":>7} {"$/day":>8}  {"mean/t":>7}')
        configs = []
        for p in [0.40, 0.50, 0.60]:
            for mfe in [10, 20, 30]:
                for sc in [5, 10, 15, 20]:
                    configs.append((p, mfe, sc))
        for p, mfe, sc in configs:
            trades = simulate(t_start, t_end, dec_models, mfe_models,
                                  p_thr=p, mfe_thr=mfe, score_thr=sc)
            s = report_stats(trades)
            if s is None or s['n'] == 0:
                continue
            print(f'{p:>6.2f}  {mfe:>7.0f}  {sc:>9.0f}  {s["n"]:>6}  '
                      f'{s["win_rate"]:>5.1f}%  {s["pf_wr"]:>+6.3f}  '
                      f'{s["per_day"]:>+7.1f}  ${s["mean_per_trade"]:>+6.2f}')
    else:
        trades = simulate(t_start, t_end, dec_models, mfe_models)
        s = report_stats(trades)
        if s:
            print(f'\n{args.name}: n={s["n"]} wr={s["win_rate"]:.1f}% PF={s["pf_wr"]:+.3f} '
                      f'total=${s["total"]:.0f} $/day=${s["per_day"]:.1f}')
        # Save trade-level CSV
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = OUT_DIR / f'v9c_{args.name}_trades.csv'
        rows = []
        for t in trades:
            pnl = ((t['exit_price']-t['entry_price'])/TICK*TICK_DOLLAR if t['side']=='LONG'
                       else (t['entry_price']-t['exit_price'])/TICK*TICK_DOLLAR)
            rows.append({
                'entry_ts': t['entry_ts'], 'exit_ts': t.get('exit_ts'),
                'side': t['side'], 'entry_price': t['entry_price'],
                'exit_price': t.get('exit_price'), 'pnl_dollars': pnl,
                'p_at_entry': t['p_at_entry'], 'mfe_pred': t['mfe_pred'],
                'score': t['score'], 'reason': t.get('reason'),
                'dur_min': t.get('dur_min'),
                'peak_ticks': t['peak'], 'worst_ticks': t['worst'],
            })
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f'  saved {len(rows)} trades → {out_csv}')

        # Histogram mode per CLAUDE.md (bin width $2)
        pnls = np.array([r['pnl_dollars'] for r in rows])
        if len(pnls) > 0:
            bin_w = 2.0
            lo = np.floor(pnls.min() / bin_w) * bin_w
            hi = np.ceil(pnls.max() / bin_w) * bin_w
            bins = np.arange(lo, hi + bin_w, bin_w)
            counts, edges = np.histogram(pnls, bins=bins)
            mode_idx = int(np.argmax(counts))
            mode_lo, mode_hi = edges[mode_idx], edges[mode_idx+1]
            mode_center = (mode_lo + mode_hi) / 2
            # bootstrap mean CI
            B = 4000; rng = np.random.default_rng(42)
            means = np.empty(B)
            for b in range(B):
                means[b] = rng.choice(pnls, size=len(pnls), replace=True).mean()
            ci_lo, ci_hi = np.percentile(means, [2.5, 97.5])
            print(f'\n  TRADE-LEVEL $ DISTRIBUTION (n={len(pnls)}):')
            print(f'    mode bin    : [${mode_lo:.0f}, ${mode_hi:.0f})  center=${mode_center:.0f}  count={counts[mode_idx]} ({100*counts[mode_idx]/len(pnls):.1f}%)')
            print(f'    median      : ${np.median(pnls):+.2f}')
            print(f'    mean (95%CI): ${pnls.mean():+.3f}  [${ci_lo:+.3f}, ${ci_hi:+.3f}]')
            print(f'    min/max     : ${pnls.min():+.0f} / ${pnls.max():+.0f}')
            # Top 10 bins for context
            top = np.argsort(counts)[::-1][:10]
            print(f'    top 10 bins :')
            for k in top:
                print(f'      [${edges[k]:+5.0f}, ${edges[k+1]:+5.0f}) : {counts[k]:5d}  {"#"*int(60*counts[k]/counts[mode_idx])}')


if __name__ == '__main__':
    main()
