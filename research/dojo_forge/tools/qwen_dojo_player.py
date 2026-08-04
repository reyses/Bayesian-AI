"""QWEN DOJO PLAYER — the taught student plays the pocket dojo (owner proposal
2026-07-28: "have qwen try the same approach, you or a sonnet teaches it, then
we see what happens").

Claude teaches via a CURRICULUM system prompt distilled from the owner's
captured process (OWNER_PROCESS.md) + the v1 backtest lessons (doc-107: no
scratch-cuts; room matters; selectivity). Qwen then plays fogged days under
the SAME engine semantics as owner_process_v1 (next-bar-open fills, target
touch, EOD flat, friction) so the three-way score is apples-to-apples:
OWNER vs QWEN vs v1.1-machine on identical days.

Per decision point (every DECIDE_EVERY bars) qwen sees a causal text frame:
recent bars, cubic value/slope/curvature, σ, telescope lines, theme cascade,
position. It answers JSON {action, target, stop, why}. Reasoning is logged —
corpus for the distillation program.

Run:  (needs CUDA libs on LD_LIBRARY_PATH)
  python research/dojo_forge/tools/qwen_dojo_player.py --n-days 10 [--seed 3]
Out:  research/dojo_forge/reports/human_dojo/qwen_player_<seed>.jsonl (decisions)
      research/dojo_forge/reports/human_dojo/qwen_player_summary.json|md
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cubic_regression as _cub                          # noqa: E402
from level_coordinate_system import telescope             # noqa: E402
from owner_process_v1 import (DATA, OUT, theme_series, FRICTION_PT, PT_USD,
                              CUBIC_W_1M, boot_ci, pf_trade_wr)  # noqa: E402

BLOB = ('/media/moi/WindowsCode/ollama/models/blobs/'
        'sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e')
N_CTX = 8192
DECIDE_EVERY = 15        # bars between decision points (~4.6s/decision budget)
WARMUP = 90
MAX_TOK = 700            # thinking model: room for <think> + JSON
BARS_SHOWN = 20          # recent bars serialized into the frame

CURRICULUM = """You are a student trader being taught a specific discretionary process for MNQ futures (1-minute bars). Follow THE PROCESS, not your own ideas.

THE PROCESS (from the master):
1. COORDINATE FRAME: horizontal reference levels (given each frame, with touch counts). They are a coordinate system, not predictions. Price moves cell-to-cell between them.
2. THEME (the "music"): the cascade of 1h/4h/day regimes is given. Trade ONLY in the theme direction. If scales conflict or all are chop, there is NO theme: do NOT trade, wait.
3. ENTRY: only with-theme, at a reference level on the correct side (support below for long, resistance above for short), when the cubic regression slope agrees with the theme. Enter as price BOUNCES from the level, not while it slices through.
4. TARGET: the NEXT reference level in the trade direction (the other cusp) — for a LONG the target must be ABOVE the current price; for a SHORT it must be BELOW. Never target the level you are entering at. If the room to target is under 10 points, SKIP the trade — friction eats small trades.
5. EXITS: NO panic exits, NO scratches. Losers cut themselves — hold to target unless the THEME flips to the opposite direction (then exit). This is proven: cut policies lose money.
6. SELECTIVITY: the master takes ~5 trades a day. Most frames are a HOLD/WAIT. When in doubt, wait.

ANSWER FORMAT: reply with ONLY a JSON object, no other text:
{"action": "hold"|"long"|"short"|"exit", "target": <price or null>, "stop": null, "why": "<one short sentence>"}
- "hold" = do nothing (also when waiting in a position).
- "long"/"short" only when FLAT and the process conditions are met; always set target to the next reference level.
- "exit" only when IN a position and the theme has flipped opposite.
/no_think"""


def strip_think(txt):
    return re.sub(r'<think>.*?</think>', '', txt, flags=re.S).strip()


def frame_text(i, o, h, l, c, cub, slp, curv, sig, th_names, lines, pos):
    rows = []
    for j in range(max(0, i - BARS_SHOWN + 1), i + 1):
        rows.append(f"{j-i:>3}: O{o[j]:.1f} H{h[j]:.1f} L{l[j]:.1f} C{c[j]:.1f}")
    lv = '\n'.join(f"  {nm} {p:.1f} ({t} touches)" for nm, p, t in lines) or '  (none yet)'
    cube = (f"value {cub[i]:.1f}, slope {slp[i]:+.2f}, curvature {curv[i]:+.3f}"
            if np.isfinite(cub[i]) else "warming up")
    ptxt = (f"IN POSITION: {pos['dir']} from {pos['entry']:.1f}, target {pos['target']:.1f}, "
            f"open P&L {(c[i]-pos['entry'])*(1 if pos['dir']=='long' else -1):+.1f}pt"
            if pos else "FLAT")
    sig_txt = f"\nSigma of residual: {sig[i]:.1f}pt" if np.isfinite(sig[i]) else ""
    return ("Recent 1m bars (index 0 = now):\n" + '\n'.join(rows) +
            f"\n\nCubic regression (7.5min): {cube}" + sig_txt +
            f"\nTheme cascade: 1h={th_names[0]} 4h={th_names[1]} day={th_names[2]}"
            f"\nReference levels:\n{lv}"
            f"\nYour state: {ptxt}"
            f"\n\nDecision?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-days', type=int, default=10)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--days', nargs='*', default=None)
    args = ap.parse_args()

    days_all = sorted(f[:-8] for f in os.listdir(DATA) if f.endswith('.parquet'))
    if args.days:
        days = args.days
    else:
        rng = np.random.default_rng(args.seed)
        days = sorted(rng.choice(days_all[5:], args.n_days, replace=False))

    from llama_cpp import Llama
    print('loading qwen3:14b ...', flush=True)
    llm = Llama(model_path=BLOB, n_gpu_layers=-1, n_ctx=N_CTX, seed=42, verbose=False)

    log_path = os.path.join(OUT, f'qwen_player_seed{args.seed}.jsonl')
    day_results = []
    for day in days:
        df = pd.read_parquet(os.path.join(DATA, f'{day}.parquet'))[
            ['timestamp', 'open', 'high', 'low', 'close']]
        o = df['open'].to_numpy(); h = df['high'].to_numpy()
        l = df['low'].to_numpy(); c = df['close'].to_numpy()
        n = len(c)
        th = theme_series(c)
        cub, slp, curv = _cub.rolling(c, CUBIC_W_1M, 60)
        sig = pd.Series(c - cub).rolling(20, min_periods=5).std().to_numpy()
        NAME = {1: 'UP', -1: 'DOWN', 0: 'chop'}

        pos = None; pending = None; pnl = 0.0; trades = 0
        lines_cache = None; lines_bar = -10**9
        decisions = 0
        for i in tqdm(range(WARMUP, n - 1, 1), desc=day, ncols=70, leave=False):
            # fills & mechanical exits every bar (same semantics as v1)
            if pending is not None:
                pos = dict(dir=pending['dir'], entry=float(o[i]),
                           target=pending['target']); pending = None
            if pos:
                d = 1 if pos['dir'] == 'long' else -1
                hit = (d > 0 and h[i] >= pos['target']) or (d < 0 and l[i] <= pos['target'])
                if hit:
                    pts = (pos['target'] - pos['entry']) * d - FRICTION_PT
                    pnl += pts; trades += 1
                    with open(log_path, 'a') as f:
                        f.write(json.dumps(dict(day=day, bar=i, event='close',
                                                reason='target', pts=round(pts, 2))) + '\n')
                    pos = None
            if (i - WARMUP) % DECIDE_EVERY:
                continue
            # refit levels sparsely
            if i - lines_bar >= DECIDE_EVERY:
                try:
                    lines_cache = [(sc['name'], L['price'], L['touches'])
                                   for sc in telescope(df.iloc[:i + 1])
                                   for L in sc['lines']]
                except Exception:
                    lines_cache = []
                lines_bar = i
            # regimes per scale for display
            th_names = []
            from owner_process_v1 import THEME_SCALES, rolling_slope
            for _, W, db in THEME_SCALES:
                if i >= W:
                    s_ = rolling_slope(c[max(0, i - W * 2):i + 1], W)[-1]
                    th_names.append('UP' if s_ > db else 'DOWN' if s_ < -db else 'chop')
                else:
                    th_names.append('warming')
            prompt = frame_text(i, o, h, l, c, cub, slp, curv, sig, th_names,
                                lines_cache, pos)
            t0 = time.time()
            try:
                r = llm.create_chat_completion(
                    messages=[{'role': 'system', 'content': CURRICULUM},
                              {'role': 'user', 'content': prompt}],
                    max_tokens=MAX_TOK, temperature=0)
                raw = strip_think(r['choices'][0]['message']['content'])
                m = re.search(r'\{.*\}', raw, flags=re.S)
                act = json.loads(m.group(0)) if m else {'action': 'hold', 'why': 'unparseable'}
            except Exception as e:
                act = {'action': 'hold', 'why': f'error {e}'}
            decisions += 1
            with open(log_path, 'a') as f:
                f.write(json.dumps(dict(day=day, bar=i, event='decide',
                                        latency=round(time.time() - t0, 1),
                                        pos=bool(pos), **{k: act.get(k) for k in
                                        ('action', 'target', 'why')})) + '\n')
            a = act.get('action')
            if a in ('long', 'short') and pos is None and pending is None:
                tgt = act.get('target')
                # target must sit on the PROFIT side (long: above, short: below)
                # with real room — a target at/behind entry would "fill" at a
                # loss instantly (seed-3 bug: qwen targeted its own support)
                ok = (isinstance(tgt, (int, float)) and
                      ((a == 'long' and tgt >= c[i] + 5.0) or
                       (a == 'short' and tgt <= c[i] - 5.0)))
                if ok:
                    pending = dict(dir=a, target=float(tgt))
                else:
                    with open(log_path, 'a') as f:
                        f.write(json.dumps(dict(day=day, bar=i,
                                                event='reject_bad_target',
                                                action=a, target=tgt)) + '\n')
            elif a == 'exit' and pos is not None:
                d = 1 if pos['dir'] == 'long' else -1
                pts = (o[i + 1] - pos['entry']) * d - FRICTION_PT
                pnl += pts; trades += 1
                with open(log_path, 'a') as f:
                    f.write(json.dumps(dict(day=day, bar=i, event='close',
                                            reason='qwen_exit', pts=round(pts, 2))) + '\n')
                pos = None
        if pos:                                   # EOD flat
            d = 1 if pos['dir'] == 'long' else -1
            pts = (c[n - 1] - pos['entry']) * d - FRICTION_PT
            pnl += pts; trades += 1
            pos = None
        day_results.append(dict(day=day, pnl_usd=round(pnl * PT_USD, 2),
                                trades=trades, decisions=decisions))
        print(f"{day}: {trades} trades, ${pnl*PT_USD:+.0f}", flush=True)

    dvals = np.array([r['pnl_usd'] for r in day_results])
    summary = dict(seed=args.seed, days=[r['day'] for r in day_results],
                   per_day=day_results,
                   total_usd=round(float(dvals.sum()), 2),
                   usd_per_day_mean=round(float(dvals.mean()), 2))
    if len(dvals) >= 5:
        lo, hi = boot_ci(dvals)
        summary['usd_per_day_ci95'] = [round(lo, 2), round(hi, 2)]
        summary['significant'] = bool(lo > 0 or hi < 0)
    # baseline comparison on the SAME days (v1.1 no-cut)
    b = os.path.join(OUT, 'owner_process_v1_1_nocut.csv')
    if os.path.exists(b):
        bb = pd.read_csv(b); bb['usd'] = bb['pts'] * PT_USD
        base = bb.groupby('day')['usd'].sum()
        summary['v1_1_same_days_usd'] = {d: round(float(base.get(d, 0.0)), 2)
                                         for d in summary['days']}
        summary['v1_1_same_days_total'] = round(float(sum(
            summary['v1_1_same_days_usd'].values())), 2)
    with open(os.path.join(OUT, 'qwen_player_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
