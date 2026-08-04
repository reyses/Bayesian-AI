"""OWNER PROCESS v1 — causal backtest of the captured discretionary process
(2026-07-28; source: reports/human_dojo/OWNER_PROCESS.md).

The three codified pieces, mechanized EXACTLY as captured — no extra cleverness:
  THEME  : TF-cascade alignment. Rolling linear-regression slope at 1h/4h/day
           scales with per-scale deadbands (green/red/gray). Theme = direction
           iff >=2 scales agree and none opposes. Gray everywhere = NO THEME =
           flat ("don't force direction in chop" — the 03:00 lesson).
  ENTRY  : with-theme only. Price touches a telescope frame-line band (the cusp
           REGION, +/-tau) on the theme side (support for long / resistance for
           short) AND the cubic turns in the theme direction (slope crosses to
           agree) -> enter at next bar open (causal).
  EXIT   : (a) TARGET — price reaches the band of the NEXT frame line in the
           trade direction ("wait until the other cusp"), exit next bar open;
           (b) SCRATCH — cubic harsh turn against the position (slope sign
           flips against with curvature also against), exit next bar open;
           (c) theme dies (goes flat/opposite), exit; (d) EOD flat.

Frame: TF telescope refit every REFIT_BARS on trailing bars (prev day + today
so far) — the frame-stability gate proved overnight-STATIC frames decay, so the
frame ROLLS. All decisions at bar close, filled next bar open. Friction charged
per round trip. Metrics per CLAUDE.md: PF-based trade WR, day WR, $/day mean +
95% bootstrap CI + significance statement.

Run:
  python research/dojo_forge/tools/owner_process_v1.py [--limit-days N]
Output -> research/dojo_forge/reports/human_dojo/owner_process_v1.{md,json,csv}
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from level_coordinate_system import DATA, OUT, telescope, TELESCOPE
import cubic_regression as _cub

# ---- constants (MNQ + captured-process knobs) ----
PT_USD = 2.0                 # MNQ $/point (tick 0.25 = $0.50)
FRICTION_PT = 0.89           # round-trip commission+slippage, points (project standard)
CUBIC_W_1M = 8               # ~7.5min cubic state on 1m bars (matches 5s window 90)
REFIT_BARS = 15              # telescope refit cadence (frame moves slowly)
TRAIL_BARS = 2800            # trailing context for the frame (~2 sessions of 1m)
WARMUP = 60                  # bars before any trading (frame + cubic warm)
# theme cascade: (scale name, slope window bars, deadband pts) — from the
# 2026-07-27 cascade session (DB 1h=6 / 4h=15 / day=25)
THEME_SCALES = [('1h', 60, 6.0), ('4h', 240, 15.0), ('day', 480, 25.0)]
# --- selectivity (v1 smoke run: 60 trades/day vs owner's ~5; gross was
#     POSITIVE pre-friction — the leak was overtrading, not direction) ---
MIN_TARGET_PT = 10.0         # a trade needs ROOM: target >= this many pts away
                             # (owner never takes a cusp-to-cusp trade whose
                             # width friction would eat; ~11x round-trip cost)
SCRATCH_SLOPE_SIG = 1.0      # "HARSH turn" = cubic slope against pos AND its
                             # magnitude > this many rolling σ of slope (sign
                             # flip alone fired 78% of exits = noise)
SLOPE_SIG_W = 60             # rolling window (bars) for the slope-σ scale
COOLDOWN_BARS = 10           # bars after an exit before re-entry (one band must
                             # not re-trigger every bar)
THEME_EXIT_OPPOSITE_ONLY = False   # v1.2 flag: True = hold through GRAY theme
                             # (the owner's "music" doesn't flicker; a gray
                             # wobble is not a theme change), exit only when the
                             # cascade flips OPPOSITE. v1.1 decomposition: theme
                             # ->0 exits were -$47/trade, mirror of target +$47.
BOOT = 4000                  # bootstrap resamples (project standard)
SEED = 11


def rolling_slope(c, W):
    """End-anchored rolling linear-regression slope, scaled to pts-per-window."""
    n = len(c); out = np.full(n, np.nan)
    x = np.arange(W); sx = x.sum(); sxx = (x * x).sum(); den = W * sxx - sx * sx
    for i in range(W - 1, n):
        y = c[i - W + 1:i + 1]
        out[i] = (W * (x * y).sum() - sx * y.sum()) / den
    return out * W


def theme_series(close):
    """Per-bar theme: +1 up / -1 down / 0 none. >=2 scales agree, none opposes."""
    n = len(close)
    regs = []
    for _, W, db in THEME_SCALES:
        s = rolling_slope(close, W)
        r = np.zeros(n, int)
        r[s > db] = 1; r[s < -db] = -1
        regs.append(r)
    regs = np.stack(regs)                      # (3, n)
    th = np.zeros(n, int)
    for i in range(n):
        v = regs[:, i]
        for d in (1, -1):
            if (v == d).sum() >= 2 and (v == -d).sum() == 0:
                th[i] = d; break
    return th


def run_day(df_prev, df_day):
    """Backtest one day causally. Returns list of trade dicts (points, gross)."""
    # trailing context = prev day + today; theme/cubic computed on the full
    # trail then indexed at today's bars (all end-anchored -> causal)
    both = pd.concat([df_prev, df_day], ignore_index=True) if df_prev is not None else df_day
    off = len(both) - len(df_day)
    c = both['close'].to_numpy(float)
    o = both['open'].to_numpy(float)
    h = both['high'].to_numpy(float); l = both['low'].to_numpy(float)
    th = theme_series(c)
    _, slp, curv = _cub.rolling(c, CUBIC_W_1M, 60)
    # rolling σ of cubic slope — the scale for "harsh"
    slp_sig = pd.Series(slp).rolling(SLOPE_SIG_W, min_periods=20).std().to_numpy()

    trades = []
    pos = 0; entry_px = 0.0; entry_i = 0; target = np.nan
    frame = []; last_fit = -10**9; last_exit_i = -10**9
    n = len(both)
    for i in range(off + WARMUP, n - 1):
        # rolling frame refit (causal: bars up to i)
        if i - last_fit >= REFIT_BARS:
            s0 = max(0, i + 1 - TRAIL_BARS)
            sub = both.iloc[s0:i + 1]
            frame = [(L['price'], sc['tau']) for sc in telescope(sub)
                     for L in sc['lines']]
            frame.sort()
            last_fit = i
        if not frame:
            continue
        px = c[i]

        if pos != 0:
            # ---- exits (decided at close i, filled open i+1) ----
            exit_reason = None
            # (a) target: reached the band of the next line in trade direction
            if np.isfinite(target) and ((pos > 0 and h[i] >= target) or
                                        (pos < 0 and l[i] <= target)):
                exit_reason = 'target'
            # (b) scratch: cubic HARSH turn against — slope against pos with
            # magnitude beyond its own recent σ, curvature also against
            elif (np.isfinite(slp[i]) and np.isfinite(curv[i]) and np.isfinite(slp_sig[i]) and
                  np.sign(slp[i]) == -pos and
                  abs(slp[i]) > SCRATCH_SLOPE_SIG * slp_sig[i] and
                  np.sign(curv[i]) == -pos):
                exit_reason = 'scratch'
            # (c) theme died or flipped (v1.2: opposite flip only — hold gray)
            elif (th[i] == -pos if THEME_EXIT_OPPOSITE_ONLY else th[i] != pos):
                exit_reason = 'theme'
            if exit_reason:
                fill = o[i + 1]
                pts = (fill - entry_px) * pos - FRICTION_PT
                trades.append(dict(entry_i=entry_i - off, exit_i=i + 1 - off,
                                   dir=pos, entry=entry_px, exit=fill,
                                   pts=pts, reason=exit_reason))
                pos = 0; last_exit_i = i
            continue

        # ---- entries ----
        t = th[i]
        if t == 0 or not np.isfinite(slp[i]) or i - last_exit_i < COOLDOWN_BARS:
            continue
        # frame line band on the theme side: support below for long,
        # resistance above for short — touched THIS bar
        cand = None
        for lp, lt in frame:
            if t > 0 and l[i] <= lp + lt and px >= lp - lt and lp <= px + lt:
                cand = (lp, lt)
            elif t < 0 and h[i] >= lp - lt and px <= lp + lt and lp >= px - lt:
                cand = (lp, lt); break
        if cand is None:
            continue
        # cubic agrees with theme direction
        if np.sign(slp[i]) != t:
            continue
        # target = next line beyond entry in trade direction
        lp0 = cand[0]
        if t > 0:
            above = [lp for lp, _ in frame if lp > lp0 + cand[1]]
            target = min(above) if above else np.nan
        else:
            below = [lp for lp, _ in frame if lp < lp0 - cand[1]]
            target = max(below) if below else np.nan
        if not np.isfinite(target) or abs(target - px) < MIN_TARGET_PT:
            continue                       # no other cusp, or no ROOM -> no trade
        pos = t; entry_px = o[i + 1]; entry_i = i + 1

    # (d) EOD flat
    if pos != 0:
        fill = c[n - 1]
        pts = (fill - entry_px) * pos - FRICTION_PT
        trades.append(dict(entry_i=entry_i - off, exit_i=n - 1 - off, dir=pos,
                           entry=entry_px, exit=fill, pts=pts, reason='eod'))
    return trades


def pf_trade_wr(pts):
    """Project-canonical Trade WR = PF - 1 (profit-factor-based, NOT count)."""
    w = pts[pts > 0].sum(); L = -pts[pts < 0].sum()
    if L <= 0:
        return np.inf if w > 0 else 0.0
    return w / L - 1.0


def boot_ci(x, stat=np.mean, n=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    bs = np.array([stat(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit-days', type=int, default=None)
    args = ap.parse_args()

    days = sorted(f[:-8] for f in os.listdir(DATA) if f.endswith('.parquet'))
    if args.limit_days:
        days = days[-args.limit_days:]
    all_trades = []; day_pnl = {}
    prev = None
    for d in tqdm(days, desc='days'):
        df = pd.read_parquet(os.path.join(DATA, f'{d}.parquet'))[
            ['timestamp', 'open', 'high', 'low', 'close']]
        tr = run_day(prev, df) if prev is not None else []
        for t in tr:
            t['day'] = d
        all_trades += tr
        day_pnl[d] = sum(t['pts'] for t in tr) * PT_USD if tr else None
        prev = df

    pts = np.array([t['pts'] for t in all_trades])
    active = {d: v for d, v in day_pnl.items() if v is not None and
              any(t['day'] == d for t in all_trades)}
    dvals = np.array(list(active.values()))
    res = {
        'n_days_run': len(days) - 1, 'n_active_days': len(dvals),
        'n_trades': len(pts),
        'trade_wr_pf': round(pf_trade_wr(pts), 3) if len(pts) else None,
        'day_wr': round(float((dvals > 0).mean()), 3) if len(dvals) else None,
        'total_usd': round(float(pts.sum() * PT_USD), 2) if len(pts) else 0.0,
        'exit_mix': {r: int(sum(1 for t in all_trades if t['reason'] == r))
                     for r in ('target', 'scratch', 'theme', 'eod')},
    }
    if len(dvals) >= 5:
        lo, hi = boot_ci(dvals)
        res['usd_per_day_mean'] = round(float(dvals.mean()), 2)
        res['usd_per_day_ci95'] = [round(lo, 2), round(hi, 2)]
        res['significant'] = bool(lo > 0 or hi < 0)
    if len(pts) >= 5:
        lo, hi = boot_ci(pts * PT_USD)
        res['usd_per_trade_mean'] = round(float(pts.mean() * PT_USD), 2)
        res['usd_per_trade_ci95'] = [round(lo, 2), round(hi, 2)]

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(all_trades).to_csv(os.path.join(OUT, 'owner_process_v1.csv'), index=False)
    with open(os.path.join(OUT, 'owner_process_v1.json'), 'w') as f:
        json.dump(res, f, indent=2)
    sig = ('SIGNIFICANT' if res.get('significant') else
           'NOT significant (CI includes 0)') if 'usd_per_day_ci95' in res else 'n/a'
    with open(os.path.join(OUT, 'owner_process_v1.md'), 'w') as f:
        f.write("# Owner process v1 — causal backtest\n\n"
                "Rules: THEME (cascade >=2 agree, none oppose) + ENTRY (frame-band "
                "touch + cubic agrees) + EXIT (other cusp / cubic harsh turn / theme "
                f"death / EOD). Friction {FRICTION_PT}pt RT.\n\n```json\n"
                + json.dumps(res, indent=2) + f"\n```\n\n$/day: {sig}\n")
    print(json.dumps(res, indent=2))
    print('$/day:', sig)


if __name__ == '__main__':
    main()
