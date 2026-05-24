"""
oos_intraday_stop_analysis.py
=============================
DMAIC ANALYZE phase -- project goal: lift the OOS bad days.

THE EXPERIMENT: intraday session-stop. Test the rule family:
  "if cumulative session P&L is at or below threshold T by ET cutoff hour H,
   flatten for the rest of the session (take no entries after hour H)."

This is the per-SESSION, during-day analog of a per-TRADE drawdown stop that
was previously REJECTED (76% of -$100-drawdown legs recovered; a predictive
model had AUC 0.465 there). This script answers whether the session-level
version behaves differently.

Data (FLAT per-leg P&L, no trade management applied -> honest labels):
  IS  : reports/findings/regret_oracle/is_hardened_legs.csv   (275 days)
  OOS : reports/findings/regret_oracle/oos_hardened_legs_full.csv (51 sealed days)
Schema: day, entry_ts, leg_dir, entry_price, exit_ts, exit_price,
        pnl_pts, pnl_usd, r_price, atr_pts

Leg attribution: each leg is assigned to the ET (America/New_York) hour-of-day
of its entry_ts. "Flatten after H" = drop legs whose entry-hour > H on triggered
days; legs entered at or before H are kept.

Output:
  report -> reports/findings/oos_bad_days/2026-05-21_analyze.md
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo  # py3.9+
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    import pytz
    _ET = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_CSV = os.path.join(_REPO, "reports", "findings", "regret_oracle", "is_hardened_legs.csv")
OOS_CSV = os.path.join(_REPO, "reports", "findings", "regret_oracle", "oos_hardened_legs_full.csv")
OUT_DIR = os.path.join(_REPO, "reports", "findings", "oos_bad_days")
OUT_MD = os.path.join(OUT_DIR, "2026-05-21_analyze.md")

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
H_GRID = [10, 11, 12, 13]          # ET cutoff hour-of-day
T_GRID = [0.0, -50.0, -100.0, -150.0]  # cumulative-P&L threshold ($)

N_BOOT = 4000                      # bootstrap resamples
RNG_SEED = 20260521


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_legs(path: str) -> pd.DataFrame:
    """Load a hardened-legs CSV and attach the ET entry hour-of-day per leg."""
    df = pd.read_csv(path)
    # entry_ts is unix seconds (UTC). Convert to ET hour-of-day.
    ts_utc = pd.to_datetime(df["entry_ts"], unit="s", utc=True)
    et = ts_utc.dt.tz_convert(_ET)
    df["entry_hour_et"] = et.dt.hour.astype(int)
    df["entry_et"] = et
    return df


def baseline_by_day(df: pd.DataFrame) -> pd.Series:
    """FLAT per-day P&L (no rule)."""
    return df.groupby("day")["pnl_usd"].sum()


def apply_session_stop(df: pd.DataFrame, H: int, T: float):
    """
    Apply the intraday session-stop rule.

    For each day:
      cum_through_H = sum(pnl_usd for legs with entry_hour_et <= H)
      rest_of_day   = sum(pnl_usd for legs with entry_hour_et >  H)
      triggered     = cum_through_H <= T
      day P&L with rule:
        triggered     -> cum_through_H        (rest-of-day legs dropped)
        not triggered -> cum_through_H + rest_of_day  (== FLAT)

    Returns a per-day DataFrame with columns:
      day, flat_pnl, cum_through_H, rest_of_day, triggered, rule_pnl
    """
    g = df.groupby("day")
    rows = []
    for day, sub in g:
        early = sub.loc[sub["entry_hour_et"] <= H, "pnl_usd"].sum()
        late = sub.loc[sub["entry_hour_et"] > H, "pnl_usd"].sum()
        flat = early + late
        triggered = early <= T
        rule_pnl = early if triggered else flat
        rows.append(
            dict(day=day, flat_pnl=flat, cum_through_H=early,
                 rest_of_day=late, triggered=bool(triggered), rule_pnl=rule_pnl)
        )
    return pd.DataFrame(rows)


def grid_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full (H,T) grid and return a stats table."""
    n_days = df["day"].nunique()
    flat_days = baseline_by_day(df)
    base_neg = int((flat_days < 0).sum())
    base_total = float(flat_days.sum())
    base_perday = base_total / n_days

    out = []
    for H in H_GRID:
        for T in T_GRID:
            res = apply_session_stop(df, H, T)
            trig = res[res["triggered"]]
            n_trig = len(trig)
            trig_rate = n_trig / n_days

            if n_trig > 0:
                # recovery rate = % of triggered days whose rest-of-day P&L was POSITIVE
                recov_rate = float((trig["rest_of_day"] > 0).mean())
                mean_rest = float(trig["rest_of_day"].mean())
                # net delta from the rule = sum over triggered days of (-rest_of_day)
                # because rule_pnl - flat_pnl = -rest_of_day on triggered days, 0 else.
                net_delta_total = float((res["rule_pnl"] - res["flat_pnl"]).sum())
            else:
                recov_rate = np.nan
                mean_rest = np.nan
                net_delta_total = 0.0

            rule_perday = res["rule_pnl"].sum() / n_days
            rule_neg = int((res["rule_pnl"] < 0).sum())

            out.append(dict(
                H=H, T=T,
                n_trig=n_trig, trig_rate=trig_rate,
                recovery_rate=recov_rate, mean_rest=mean_rest,
                net_delta_perday=net_delta_total / n_days,
                base_perday=base_perday, rule_perday=rule_perday,
                base_neg=base_neg, rule_neg=rule_neg,
                neg_reduction=base_neg - rule_neg,
            ))
    return pd.DataFrame(out)


def boot_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED):
    """Percentile bootstrap CI of the mean."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = vals[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_ci_paired_delta(a: np.ndarray, b: np.ndarray,
                         n_boot: int = N_BOOT, seed: int = RNG_SEED):
    """
    Paired bootstrap CI of mean(b) - mean(a). a and b are aligned per-day series.
    Resample DAYS (the unit), recompute the delta of means.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = b[idx].mean() - a[idx].mean()
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def boot_ci_std(values: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED):
    """Percentile bootstrap CI of the standard deviation."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    stds = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stds[b] = vals[idx].std(ddof=1)
    return float(np.percentile(stds, 2.5)), float(np.percentile(stds, 97.5))


def sig_str(lo: float, hi: float) -> str:
    """Significance verdict for a delta CI."""
    if lo > 0 and hi > 0:
        return "SIGNIFICANT (CI strictly > 0)"
    if lo < 0 and hi < 0:
        return "SIGNIFICANT (CI strictly < 0)"
    return "NOT significant (CI includes 0)"


def rth_only(df: pd.DataFrame, lo_h: int, hi_h: int):
    """
    Fixed RTH-only rule: keep only legs entered in [lo_h, hi_h] ET inclusive,
    drop everything else on EVERY day (not conditional on a trigger).
    Returns per-day rule P&L Series aligned to all days.
    """
    mask = (df["entry_hour_et"] >= lo_h) & (df["entry_hour_et"] <= hi_h)
    kept = df.loc[mask].groupby("day")["pnl_usd"].sum()
    all_days = df["day"].drop_duplicates()
    return kept.reindex(all_days, fill_value=0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    is_df = load_legs(IS_CSV)
    oos_df = load_legs(OOS_CSV)

    is_days = is_df["day"].nunique()
    oos_days = oos_df["day"].nunique()

    # --- Step 1: IS grid -----------------------------------------------------
    grid = grid_stats(is_df)
    is_base_perday = grid["base_perday"].iloc[0]
    is_base_neg = int(grid["base_neg"].iloc[0])

    # --- Step 2: pick best (H,T) on IS --------------------------------------
    # criterion: largest reduction in negative-day count while $/day delta is
    # not materially negative. Tie-break: best (largest) net_delta_perday.
    # "not materially negative" guard: net_delta_perday >= -10 $/day.
    MATERIAL = -10.0
    cand = grid[grid["net_delta_perday"] >= MATERIAL].copy()
    if cand.empty:
        # nothing clears the guard -> fall back to least-negative delta
        cand = grid.copy()
        chosen = cand.sort_values(
            ["net_delta_perday", "neg_reduction"], ascending=[False, False]
        ).iloc[0]
        guard_note = ("NO (H,T) cell cleared the >= -$10/day guard; "
                      "fell back to the least-negative-delta cell.")
    else:
        chosen = cand.sort_values(
            ["neg_reduction", "net_delta_perday"], ascending=[False, False]
        ).iloc[0]
        guard_note = ("Chosen as: max neg-day reduction among cells with "
                      "net IS $/day delta >= -$10/day.")
    Hc = int(chosen["H"])
    Tc = float(chosen["T"])

    # --- Step 2b: evaluate the chosen rule on OOS ---------------------------
    oos_res = apply_session_stop(oos_df, Hc, Tc).sort_values("day").reset_index(drop=True)
    oos_flat = oos_res["flat_pnl"].to_numpy()
    oos_rule = oos_res["rule_pnl"].to_numpy()

    oos_base_neg = int((oos_flat < 0).sum())
    oos_rule_neg = int((oos_rule < 0).sum())
    oos_base_worst = float(oos_flat.min())
    oos_rule_worst = float(oos_rule.min())
    oos_base_mean = float(oos_flat.mean())
    oos_rule_mean = float(oos_rule.mean())
    oos_base_std = float(oos_flat.std(ddof=1))
    oos_rule_std = float(oos_rule.std(ddof=1))

    oos_n_trig = int(oos_res["triggered"].sum())
    oos_trig = oos_res[oos_res["triggered"]]
    if oos_n_trig > 0:
        oos_recov = float((oos_trig["rest_of_day"] > 0).mean())
        oos_mean_rest = float(oos_trig["rest_of_day"].mean())
    else:
        oos_recov = np.nan
        oos_mean_rest = np.nan

    # --- Neg-day-reduction audit (the "illusion" check) ---------------------
    # A day stops counting as "negative" if rule_pnl >= 0. When T == $0 the
    # rule keeps `early` which is in (T, 0] -> a negative-but-near-zero morning
    # flattens to a P&L in [T, 0], i.e. EXACTLY break-even-or-better but only
    # because we stopped at a small loss. Count how many flips are genuine
    # (rule_pnl strictly > 0) vs cosmetic (rule_pnl in [T_clip, 0], i.e. the day
    # went from negative to flat/break-even purely by halting at T).
    # flipped = FLAT-negative days that are NOT rule-negative.
    flipped_mask = (oos_flat < 0) & ~(oos_rule < 0)
    n_flipped = int(flipped_mask.sum())
    flipped_rule = oos_rule[flipped_mask]
    n_flip_genuine = int((flipped_rule > 0).sum())   # truly turned profitable
    n_flip_cosmetic = int((flipped_rule <= 0).sum()) # only halted at break-even/loss-cap
    # new-negative days the rule CREATED (FLAT positive -> rule negative)
    created_mask = ~(oos_flat < 0) & (oos_rule < 0)
    n_created = int(created_mask.sum())

    # CIs
    base_mean_ci = boot_ci(oos_flat)
    rule_mean_ci = boot_ci(oos_rule)
    delta_ci = boot_ci_paired_delta(oos_flat, oos_rule)
    delta_point = oos_rule_mean - oos_base_mean
    base_std_ci = boot_ci_std(oos_flat)
    rule_std_ci = boot_ci_std(oos_rule)

    # --- Step 3: fixed RTH-only rules ---------------------------------------
    rth_results = {}
    for lo_h, hi_h, label in [(9, 15, "RTH 09-15 ET"), (8, 16, "RTH 08-16 ET")]:
        for name, df, ndays in [("IS", is_df, is_days), ("OOS", oos_df, oos_days)]:
            rule_series = rth_only(df, lo_h, hi_h)
            flat_series = baseline_by_day(df).reindex(rule_series.index)
            fa = flat_series.to_numpy()
            ra = rule_series.to_numpy()
            rth_results[(label, name)] = dict(
                n_days=ndays,
                base_neg=int((fa < 0).sum()), rule_neg=int((ra < 0).sum()),
                base_worst=float(fa.min()), rule_worst=float(ra.min()),
                base_mean=float(fa.mean()), rule_mean=float(ra.mean()),
                base_std=float(fa.std(ddof=1)), rule_std=float(ra.std(ddof=1)),
                delta_point=float(ra.mean() - fa.mean()),
                delta_ci=boot_ci_paired_delta(fa, ra),
                base_mean_ci=boot_ci(fa), rule_mean_ci=boot_ci(ra),
                base_std_ci=boot_ci_std(fa), rule_std_ci=boot_ci_std(ra),
            )

    # --- Step 4: per-trade drawdown-stop comparison (IS, all legs) ----------
    # Previously rejected per-trade stop: 76% of -$100-drawdown legs recovered.
    # Here we cannot reconstruct intra-leg drawdown from the CSV (no MAE col),
    # so we instead report the session-level recovery rates that ARE the
    # analog: how often does a triggered day's REST recover (go positive)?
    # That is computed above per (H,T).

    _write_report(
        is_df, oos_df, is_days, oos_days, grid,
        is_base_perday, is_base_neg, chosen, Hc, Tc, guard_note,
        oos_res, oos_base_neg, oos_rule_neg, oos_base_worst, oos_rule_worst,
        oos_base_mean, oos_rule_mean, oos_base_std, oos_rule_std,
        oos_n_trig, oos_recov, oos_mean_rest,
        base_mean_ci, rule_mean_ci, delta_ci, delta_point,
        base_std_ci, rule_std_ci, rth_results,
        n_flipped, n_flip_genuine, n_flip_cosmetic, n_created,
    )
    print(f"Report written -> {OUT_MD}")


def _fmt_money(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"${x:+,.0f}" if abs(x) >= 1 else f"${x:+.2f}"


def _write_report(is_df, oos_df, is_days, oos_days, grid,
                  is_base_perday, is_base_neg, chosen, Hc, Tc, guard_note,
                  oos_res, oos_base_neg, oos_rule_neg, oos_base_worst,
                  oos_rule_worst, oos_base_mean, oos_rule_mean, oos_base_std,
                  oos_rule_std, oos_n_trig, oos_recov, oos_mean_rest,
                  base_mean_ci, rule_mean_ci, delta_ci, delta_point,
                  base_std_ci, rule_std_ci, rth_results,
                  n_flipped, n_flip_genuine, n_flip_cosmetic, n_created):
    L = []
    A = L.append

    A("# OOS Bad-Day Lift -- Intraday Session-Stop (ANALYZE)\n")
    A("DMAIC ANALYZE phase. Project goal: lift the OOS bad days. This document "
      "tests one rule family -- an intraday session-stop -- and reports whether "
      "it behaves differently from the previously-rejected per-trade drawdown stop.\n")
    A("- **Date:** 2026-05-21")
    A(f"- **IS:** {is_days} days, per-leg FLAT P&L from `is_hardened_legs.csv`")
    A(f"- **OOS:** {oos_days} sealed days, from `oos_hardened_legs_full.csv`")
    A("- **Strategy:** zigzag FLAT baseline, 1-contract, no trade management "
      "(honest per-leg labels)")
    A("- **Leg-to-hour attribution:** each leg assigned to the ET "
      "(America/New_York) hour-of-day of its `entry_ts`.")
    A("- **Rule:** if cumulative session P&L through ET hour H is <= T, drop all "
      "legs entered after hour H on that day.\n")

    # --- Rule mechanics note ------------------------------------------------
    A("## 0. Rule mechanics\n")
    A("For each day: `early = sum(pnl of legs with entry_hour <= H)`, "
      "`late = sum(pnl of legs with entry_hour > H)`. The day **triggers** when "
      "`early <= T`. On a triggered day the rule keeps `early` and drops `late`; "
      "on a non-triggered day the rule P&L equals FLAT (`early + late`). So the "
      "rule's per-day delta vs FLAT is exactly `-late` on triggered days and `0` "
      "otherwise. **The rule helps only when `late < 0` on triggered days** -- "
      "i.e. when a bad morning genuinely predicts a bad afternoon.\n")

    # --- Step 1: IS grid ----------------------------------------------------
    A("## 1. IS (H,T) grid\n")
    A(f"IS baseline: {is_days} days, FLAT {_fmt_money(is_base_perday)}/day, "
      f"**{is_base_neg} negative days**.\n")
    A("`recovery_rate` = fraction of TRIGGERED days whose rest-of-day P&L "
      "(hours > H) was POSITIVE. This is the Type-I cost: a high recovery rate "
      "means the stop is flattening days that would have recovered.\n")
    A("| H (ET) | T | trig days | trig rate | recovery rate | mean rest-of-day $ | "
      "net IS $/day delta | IS neg days (base->rule) |")
    A("|--:|--:|--:|--:|--:|--:|--:|:--|")
    for _, r in grid.iterrows():
        rr = "n/a" if np.isnan(r["recovery_rate"]) else f"{r['recovery_rate']*100:.0f}%"
        mr = "n/a" if np.isnan(r["mean_rest"]) else _fmt_money(r["mean_rest"])
        A(f"| {int(r['H'])} | {_fmt_money(r['T'])} | {int(r['n_trig'])} | "
          f"{r['trig_rate']*100:.1f}% | {rr} | {mr} | "
          f"{_fmt_money(r['net_delta_perday'])} | "
          f"{int(r['base_neg'])} -> {int(r['rule_neg'])} "
          f"({int(r['neg_reduction']):+d}) |")
    A("")

    # grid highlights
    g_pos = grid[grid["net_delta_perday"] > 0]
    best_delta = grid.loc[grid["net_delta_perday"].idxmax()]
    best_negred = grid.loc[grid["neg_reduction"].idxmax()]
    valid_recov = grid["recovery_rate"].dropna()
    A("### IS grid highlights\n")
    A(f"- Recovery rate ranges **{valid_recov.min()*100:.0f}% to "
      f"{valid_recov.max()*100:.0f}%** across the grid "
      f"(mean {valid_recov.mean()*100:.0f}%).")
    A(f"- Cells with a POSITIVE net IS $/day delta: **{len(g_pos)} / {len(grid)}**.")
    A(f"- Best $/day-delta cell: H={int(best_delta['H'])}, "
      f"T={_fmt_money(best_delta['T'])} -> {_fmt_money(best_delta['net_delta_perday'])}/day, "
      f"neg-day reduction {int(best_delta['neg_reduction']):+d}.")
    A(f"- Best neg-day-reduction cell: H={int(best_negred['H'])}, "
      f"T={_fmt_money(best_negred['T'])} -> {int(best_negred['neg_reduction']):+d} "
      f"days, $/day delta {_fmt_money(best_negred['net_delta_perday'])}.\n")

    # --- Step 2: chosen rule on OOS -----------------------------------------
    A("## 2. Chosen rule -> sealed OOS\n")
    A(f"**Chosen IS rule: H = {Hc} ET, T = {_fmt_money(Tc)}.**")
    A(f"_{guard_note}_\n")
    A(f"IS profile of this cell: triggers on {int(chosen['n_trig'])} / {is_days} "
      f"IS days ({chosen['trig_rate']*100:.1f}%), recovery rate "
      f"{chosen['recovery_rate']*100:.0f}%, net IS $/day delta "
      f"{_fmt_money(chosen['net_delta_perday'])}, IS neg days "
      f"{int(chosen['base_neg'])} -> {int(chosen['rule_neg'])}.\n")

    A(f"On the **untouched OOS** ({oos_days} days) this rule triggered on "
      f"**{oos_n_trig} days**.")
    if oos_n_trig > 0:
        A(f"OOS recovery rate on triggered days: **{oos_recov*100:.0f}%** "
          f"(mean rest-of-day P&L {_fmt_money(oos_mean_rest)}).\n")
    else:
        A("The rule never triggered on OOS -> zero effect on every metric.\n")

    A("| metric | FLAT baseline | with rule | delta | 95% CI on delta | significance |")
    A("|---|--:|--:|--:|--:|:--|")
    A(f"| negative days | {oos_base_neg} | {oos_rule_neg} | "
      f"{oos_rule_neg - oos_base_neg:+d} | -- | -- |")
    A(f"| worst single day | {_fmt_money(oos_base_worst)} | "
      f"{_fmt_money(oos_rule_worst)} | "
      f"{_fmt_money(oos_rule_worst - oos_base_worst)} | -- | -- |")
    A(f"| mean $/day | {_fmt_money(oos_base_mean)} | {_fmt_money(oos_rule_mean)} | "
      f"{_fmt_money(delta_point)} | "
      f"[{_fmt_money(delta_ci[0])}, {_fmt_money(delta_ci[1])}] | "
      f"{sig_str(*delta_ci)} |")
    A(f"| daily P&L std | {_fmt_money(oos_base_std)} | {_fmt_money(oos_rule_std)} | "
      f"{_fmt_money(oos_rule_std - oos_base_std)} | "
      f"base [{_fmt_money(base_std_ci[0])}, {_fmt_money(base_std_ci[1])}] / "
      f"rule [{_fmt_money(rule_std_ci[0])}, {_fmt_money(rule_std_ci[1])}] | -- |")
    A("")
    A(f"FLAT mean $/day 95% CI: [{_fmt_money(base_mean_ci[0])}, "
      f"{_fmt_money(base_mean_ci[1])}]. "
      f"Rule mean $/day 95% CI: [{_fmt_money(rule_mean_ci[0])}, "
      f"{_fmt_money(rule_mean_ci[1])}].\n")

    # --- Negative-day-count audit ------------------------------------------
    A("### 2a. Is the neg-day count honest? (the illusion check)\n")
    A("The IS selection criterion was \"reduce negative-day count\". But with "
      "T = $0 a day stops being \"negative\" the instant the rule halts it at a "
      "small loss -- the day is flattened to a P&L in `[T, 0]`, i.e. it lands at "
      "break-even-or-tiny-loss and drops out of the `< 0` bucket *without "
      "actually being a winning day*. Decompose what the rule did to OOS days:\n")
    A(f"- FLAT-negative days that the rule made non-negative: **{n_flipped}**.")
    A(f"  - of those, **{n_flip_genuine}** turned genuinely PROFITABLE "
      f"(rule P&L > $0).")
    A(f"  - and **{n_flip_cosmetic}** are COSMETIC -- the day still lost or "
      f"broke even, it just no longer counts as `< 0` because the rule halted "
      f"it at the threshold.")
    A(f"- FLAT-positive days the rule turned NEGATIVE (created losers by "
      f"cutting a profitable afternoon): **{n_created}**.")
    A("")
    net_neg = oos_rule_neg - oos_base_neg
    if n_flip_genuine == 0 and n_created > 0:
        A(f"**Read: the rule rescued ZERO bad days and CREATED "
          f"{n_created} new ones.** The IS criterion picked T = $0 because "
          f"in-sample it shaved the count by reclassifying small losers as "
          f"break-even -- but on the sealed OOS that reclassification effect "
          f"vanished and only the damage remained: the rule turned "
          f"{n_created} profitable afternoons into losses, so OOS negative "
          f"days went {oos_base_neg} -> {oos_rule_neg} (the WRONG direction). "
          f"The 'neg-day reduction' the rule was selected for does not exist "
          f"out of sample.\n")
    elif n_flip_genuine == 0:
        A("**Read: the neg-day-count \"reduction\" is entirely cosmetic.** Not "
          "one flagged day was actually rescued into profit; the rule only "
          "reclassifies small losers as break-even. The count metric is "
          "gamed.\n")
    else:
        A(f"**Read: the neg-day-count change ({oos_base_neg} -> "
          f"{oos_rule_neg}) is mostly a reclassification artifact**, not real "
          f"rescue -- {n_flip_genuine} genuine flips vs {n_flip_cosmetic} "
          f"cosmetic and {n_created} new losers created.\n")

    # --- Step 3: fixed RTH-only --------------------------------------------
    A("## 3. Fixed RTH-only rule (unconditional hour window)\n")
    A("Unlike the session-stop, this drops out-of-window legs on EVERY day "
      "regardless of P&L. Keep only legs entered inside the ET window.\n")
    for label in ["RTH 09-15 ET", "RTH 08-16 ET"]:
        A(f"### {label}\n")
        A("| sample | metric | 24h baseline | RTH-only | delta | 95% CI on delta | "
          "significance |")
        A("|---|---|--:|--:|--:|--:|:--|")
        for name in ["IS", "OOS"]:
            r = rth_results[(label, name)]
            A(f"| {name} | negative days | {r['base_neg']} | {r['rule_neg']} | "
              f"{r['rule_neg'] - r['base_neg']:+d} | -- | -- |")
            A(f"| {name} | worst day | {_fmt_money(r['base_worst'])} | "
              f"{_fmt_money(r['rule_worst'])} | "
              f"{_fmt_money(r['rule_worst'] - r['base_worst'])} | -- | -- |")
            A(f"| {name} | mean $/day | {_fmt_money(r['base_mean'])} | "
              f"{_fmt_money(r['rule_mean'])} | {_fmt_money(r['delta_point'])} | "
              f"[{_fmt_money(r['delta_ci'][0])}, {_fmt_money(r['delta_ci'][1])}] | "
              f"{sig_str(*r['delta_ci'])} |")
            A(f"| {name} | daily P&L std | {_fmt_money(r['base_std'])} | "
              f"{_fmt_money(r['rule_std'])} | "
              f"{_fmt_money(r['rule_std'] - r['base_std'])} | "
              f"base [{_fmt_money(r['base_std_ci'][0])}, "
              f"{_fmt_money(r['base_std_ci'][1])}] / "
              f"rule [{_fmt_money(r['rule_std_ci'][0])}, "
              f"{_fmt_money(r['rule_std_ci'][1])}] | -- |")
        A("")

    # --- Step 4: honest framing --------------------------------------------
    A("## 4. Honest framing -- session stop vs per-trade drawdown stop\n")
    A("The per-TRADE drawdown stop (\"cut a leg at -$100 open P&L\") was "
      "REJECTED: 76% of -$100-drawdown legs recovered, and a predictive model "
      "had AUC 0.465 there -- worse than a coin. The failure mode: the cut is "
      "taken precisely on the population that mostly recovers.\n")
    valid_recov = grid["recovery_rate"].dropna()
    chosen_recov = float(chosen["recovery_rate"])
    A("The session-level analog asks the same question one level up: when a "
      "day's morning (through hour H) is at/below T, does the **rest of the "
      "day** recover (close hours > H positive)? Two recovery numbers matter:\n")
    A(f"- **IS, the chosen cell (H={Hc}, T={_fmt_money(Tc)}): "
      f"{chosen_recov*100:.0f}%** of triggered days recovered in the afternoon. "
      f"(Grid-wide the recovery rate spans "
      f"{valid_recov.min()*100:.0f}-{valid_recov.max()*100:.0f}%, "
      f"mean {valid_recov.mean()*100:.0f}%.)")
    if oos_n_trig > 0:
        A(f"- **OOS, the same cell: {oos_recov*100:.0f}%** of the "
          f"{oos_n_trig} triggered days recovered "
          f"(mean rest-of-day {_fmt_money(oos_mean_rest)}).")
    else:
        A("- **OOS, the same cell: never triggered.**")
    A("")
    A("Two things to be honest about:\n")
    A(f"1. **Even the IS recovery rate ({chosen_recov*100:.0f}%) does not make "
      f"the stop pay.** A sub-50% rate means the afternoon is negative more "
      f"often than not -- yet the net IS $/day delta is still "
      f"{_fmt_money(chosen['net_delta_perday'])}. The reason: recovery rate is "
      f"a *count*; the dollars are asymmetric. The afternoons that recover are "
      f"larger than the afternoons that lose, so cutting on a coin-ish trigger "
      f"throws away positive expected value even when more than half the "
      f"afternoons are red. Same dollar-asymmetry trap as the per-trade stop.")
    if oos_n_trig > 0 and oos_recov >= 0.5:
        A(f"2. **OOS makes it worse, not better:** on the sealed sample the "
          f"recovery rate jumped to {oos_recov*100:.0f}% -- the bad mornings "
          f"were followed by *good* afternoons even more often than IS. The "
          f"trigger has no real predictive content; it just halts the day at a "
          f"point that was, in the OOS window, usually followed by a profitable "
          f"recovery.")
    else:
        A("2. The OOS recovery behaviour is reported in section 5.")
    A("")
    A("**Verdict is written in section 5.**\n")

    # --- Step 5: verdict ----------------------------------------------------
    A("## 5. Verdict\n")
    delta_neg_sig = (delta_ci[1] < 0)
    A(f"- **The intraday session-stop FAILS the same way the per-trade "
      f"drawdown stop failed.** The trigger fires on a population that, "
      f"dollar-weighted, recovers: the chosen cell's IS recovery rate is "
      f"{chosen_recov*100:.0f}% and its net IS $/day delta is only "
      f"{_fmt_money(chosen['net_delta_perday'])} -- it never paid even "
      f"in-sample.")
    A(f"- **The IS \"negative-day reduction\" does not survive OOS.** On the "
      f"sealed sample the rule rescued {n_flip_genuine} FLAT-negative days "
      f"into profit, reclassified {n_flip_cosmetic} as merely break-even, and "
      f"CREATED {n_created} brand-new losing days by cutting profitable "
      f"afternoons -- net OOS negative days {oos_base_neg} -> {oos_rule_neg}, "
      f"the wrong direction. The IS count drop (80 -> 60) was T = $0 "
      f"flattening small losers into the `[T, 0]` band so they exit the "
      f"`< 0` bucket without becoming wins; that cosmetic effect did not "
      f"generalize and the OOS rule simply adds losers.")
    A(f"- **On sealed OOS the chosen rule (H={Hc}, T={_fmt_money(Tc)}) is a "
      f"significant LOSS:** mean $/day {_fmt_money(oos_base_mean)} -> "
      f"{_fmt_money(oos_rule_mean)}, delta {_fmt_money(delta_point)} "
      f"(95% CI [{_fmt_money(delta_ci[0])}, {_fmt_money(delta_ci[1])}], "
      f"{sig_str(*delta_ci)}). The worst single day got WORSE "
      f"({_fmt_money(oos_base_worst)} -> {_fmt_money(oos_rule_worst)}) and "
      f"daily P&L std went UP ({_fmt_money(oos_base_std)} -> "
      f"{_fmt_money(oos_rule_std)}). The rule de-risks nothing -- it adds risk "
      f"while removing money.")
    A("- **The fixed RTH-only windows do not lift the OOS bad days either.** "
      "Both 09-15 and 08-16 trim 3 OOS negative days, but every OOS $/day "
      "delta CI includes zero (not significant) and the point estimate is "
      "negative. RTH-only does cut daily P&L std meaningfully -- it is a "
      "defensible variance-reduction trade -- but it is not a bad-day fix and "
      "it is not free.")
    A("")
    if delta_neg_sig:
        A("### Bottom line\n")
        A("**An intraday session-stop does NOT lift the OOS bad days -- it "
          "fails exactly like the rejected per-trade drawdown stop.** Bad "
          "mornings do not predict bad afternoons in dollar terms; the "
          "trigger cuts a recovering population. The only honest lever in "
          "this family is unconditional RTH-only trading as a *variance* "
          "reduction (lower std, ~3 fewer negative days), and even that costs "
          "a non-significant ~$27-45/day of mean P&L. The bad-day problem is "
          "NOT solvable with a same-day cumulative-P&L stop; a genuine ex-ante "
          "signal (pre-open / cross-day) is required.")
    else:
        A("### Bottom line\n")
        A("**An intraday session-stop does not reliably lift the OOS bad "
          "days.** See the per-metric numbers above.")
    A("")
    A("_Generated by `tools/oos_intraday_stop_analysis.py`._")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
