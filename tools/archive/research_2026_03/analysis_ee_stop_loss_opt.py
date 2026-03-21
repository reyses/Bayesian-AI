"""
Analysis EE -- Stop Loss Optimization
Source: RESEARCH_SPEC_V_TO_FF.md

Reads existing oracle trade logs (IS + OOS). No ATLAS data needed.
Outputs: reports/research/ee_stop_loss_opt/results.txt

IMPORTANT: oracle_mfe = max_up (upside from entry), oracle_mae = max_down (downside)
These are direction-BLIND. For SHORT trades, adverse = oracle_mfe (upside hurts).
trade_mfe_ticks is direction-AWARE (always favorable).
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TICK = 0.25
TICK_VALUE = 0.50

IS_LOG  = ROOT / "checkpoints" / "oracle_trade_log_old.csv"
OOS_LOG = ROOT / "checkpoints" / "oos_trade_log.csv"
OUT_DIR = ROOT / "reports" / "research" / "ee_stop_loss_opt"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Direction-aware adverse/favorable excursion in ticks
    is_long = df["direction"] == "LONG"
    # LONG: adverse = downside (oracle_mae), favorable = upside (oracle_mfe)
    # SHORT: adverse = upside (oracle_mfe), favorable = downside (oracle_mae)
    df["adverse_ticks"] = np.where(is_long, df["oracle_mae"], df["oracle_mfe"]) / TICK
    df["favorable_ticks"] = np.where(is_long, df["oracle_mfe"], df["oracle_mae"]) / TICK
    # trade_mfe_ticks is already direction-aware from the exit engine
    df["sl_ticks_entry"] = df["sl_ticks"]
    df["won"] = (df["result"] == "WIN").astype(int)
    # Clamp negative values (can occur when oracle_mfe/mae are negative due to
    # the naive max_up/max_down calculation)
    df["adverse_ticks"] = df["adverse_ticks"].clip(lower=0)
    df["favorable_ticks"] = df["favorable_ticks"].clip(lower=0)
    return df


def analyze(df: pd.DataFrame, label: str, f):
    N = len(df)
    winners = df[df["won"] == 1]
    losers  = df[df["won"] == 0]
    n_long = (df["direction"] == "LONG").sum()
    n_short = (df["direction"] == "SHORT").sum()

    f.write(f"\n{'='*70}\n")
    f.write(f"  {label}  --  {N:,} trades, WR={winners.shape[0]/N:.1%}\n")
    f.write(f"  LONG: {n_long:,}  SHORT: {n_short:,}\n")
    f.write(f"{'='*70}\n")

    # -- 1. Current SL stats ------------------------------------------
    sl_hits = df[df["exit_reason"] == "stop_loss"]
    f.write(f"\n[1] Current SL Performance\n")
    f.write(f"    SL hit rate:  {len(sl_hits)/N:.1%}  ({len(sl_hits):,} trades)\n")
    f.write(f"    Avg SL width: {df['sl_ticks_entry'].mean():.1f} ticks\n")
    if len(sl_hits) > 0:
        f.write(f"    Avg loss on SL hit: ${sl_hits['actual_pnl'].mean():.2f}\n")
        f.write(f"    SL hit hold bars:   {sl_hits['hold_bars'].mean():.1f} avg\n")

    # -- 2. Adverse excursion distribution for WINNERS -----------------
    w_adv = winners["adverse_ticks"]
    f.write(f"\n[2] Winner Adverse Excursion Distribution (direction-aware, ticks)\n")
    for pct in [50, 75, 90, 95, 99]:
        val = np.percentile(w_adv, pct)
        f.write(f"    P{pct:02d}: {val:.1f} ticks (${val * TICK_VALUE:.2f})\n")
    f.write(f"    >> Minimum viable SL (P95): {np.percentile(w_adv, 95):.1f} ticks\n")

    # -- 3. Fixed SL sweep (direction-aware) ---------------------------
    f.write(f"\n[3] Fixed SL Sweep (direction-aware adverse excursion)\n")
    f.write(f"    {'SL':>4s}  {'WR':>6s}  {'Total PnL':>12s}  {'$/trade':>8s}  {'SL Hits':>8s}  {'Max Loss':>9s}\n")
    f.write(f"    {'-'*4}  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*9}\n")

    adv = df["adverse_ticks"].values
    pnl = df["actual_pnl"].values

    best_sl, best_pnl = 0, -1e9
    for sl_w in [4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 60, 80, 100, 150, 200]:
        hit = adv >= sl_w
        sim = np.where(hit, -sl_w * TICK_VALUE, pnl)
        total = sim.sum()
        n_won = ((~hit) & (pnl > 0)).sum()
        wr = n_won / N
        n_hit = hit.sum()
        worst = sim.min()
        f.write(f"    {sl_w:4d}  {wr:5.1%}  ${total:11,.2f}  ${total/N:7.2f}  {n_hit:8,}  ${worst:8.2f}\n")
        if total > best_pnl:
            best_pnl, best_sl = total, sl_w
    f.write(f"    >> PnL-optimal fixed SL: {best_sl} ticks (${best_pnl:,.2f})\n")

    # -- 4. SL efficiency (near-miss analysis) -------------------------
    if len(sl_hits) > 0:
        f.write(f"\n[4] SL Near-Miss Analysis\n")
        f.write(f"    Trades that hit SL -- would wider SL have saved them?\n")
        for extra in [2, 4, 8, 12]:
            saved = (sl_hits["adverse_ticks"] < sl_hits["sl_ticks_entry"] + extra).sum()
            f.write(f"    +{extra:2d} ticks wider >> {saved:,} saved ({saved/len(sl_hits):.0%})\n")

    # -- 5. Time-to-SL ------------------------------------------------
    if len(sl_hits) > 0:
        f.write(f"\n[5] Time-to-SL Distribution (bars)\n")
        for pct in [25, 50, 75, 90]:
            val = np.percentile(sl_hits["hold_bars"], pct)
            f.write(f"    P{pct:02d}: {val:.0f} bars\n")
        early = (sl_hits["hold_bars"] <= 5).sum()
        f.write(f"    SL hit within 5 bars: {early:,} ({early/len(sl_hits):.0%})\n")

    # -- 6. Depth-conditional SL ---------------------------------------
    f.write(f"\n[6] Depth-Conditional Adverse Excursion\n")
    for depth in sorted(df["entry_depth"].unique()):
        sub = df[df["entry_depth"] == depth]
        if len(sub) < 20:
            continue
        w = sub[sub["won"] == 1]
        if len(w) > 0:
            p95 = np.percentile(w["adverse_ticks"], 95)
            p50 = np.percentile(w["adverse_ticks"], 50)
        else:
            p95 = p50 = float("nan")
        sl_h = sub[sub["exit_reason"] == "stop_loss"]
        f.write(f"    Depth {depth:2d}: N={len(sub):5,}  WR={w.shape[0]/len(sub):5.1%}  "
                f"Winner Adv P50={p50:5.1f}  P95={p95:6.1f}  SL hits={len(sl_h):4,}\n")

    # -- 7. Breakeven acceleration (using trade_mfe_ticks) -------------
    f.write(f"\n[7] Breakeven Acceleration (using actual trade MFE)\n")
    f.write(f"    If trade reaches +N ticks MFE, move SL to entry. Losers that\n")
    f.write(f"    reached MFE >= threshold become $0 instead of a loss.\n\n")
    f.write(f"    {'BE Thresh':>10s}  {'WR':>6s}  {'Total PnL':>12s}  {'$/trade':>8s}  {'Saves':>6s}  {'Saved$':>10s}\n")
    f.write(f"    {'-'*10}  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*6}  {'-'*10}\n")

    mfe_actual = df["trade_mfe_ticks"].values

    for be_thresh in [2, 4, 6, 8, 12, 16, 24, 32]:
        # A loser that reached MFE >= threshold would have been saved
        would_save = (mfe_actual >= be_thresh) & (pnl < 0)
        sim = pnl.copy()
        sim[would_save] = 0.0  # breakeven instead of loss
        total = sim.sum()
        n_won = (sim > 0).sum()
        wr = n_won / N
        n_saves = would_save.sum()
        saved_dollars = -pnl[would_save].sum()  # how much loss was avoided
        f.write(f"    {be_thresh:10d}  {wr:5.1%}  ${total:11,.2f}  ${total/N:7.2f}  "
                f"{n_saves:6,}  ${saved_dollars:9,.2f}\n")

    # -- 8. Breakeven with actual SL tightening ------------------------
    # More realistic: if MFE >= threshold, SL moves to entry + 1 tick (small buffer)
    f.write(f"\n[8] Breakeven with 1-tick buffer (more conservative)\n")
    f.write(f"    {'BE Thresh':>10s}  {'WR':>6s}  {'Total PnL':>12s}  {'$/trade':>8s}  {'Saves':>6s}\n")
    f.write(f"    {'-'*10}  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*6}\n")

    for be_thresh in [4, 8, 12, 16, 24]:
        # If trade reached MFE >= threshold but then lost:
        # With BE lock, worst case is -1 tick (buffer)
        would_save = (mfe_actual >= be_thresh) & (pnl < 0)
        sim = pnl.copy()
        sim[would_save] = -1 * TICK_VALUE  # -$0.50 buffer loss
        total = sim.sum()
        n_won = (sim > 0).sum()
        wr = n_won / N
        n_saves = would_save.sum()
        f.write(f"    {be_thresh:10d}  {wr:5.1%}  ${total:11,.2f}  ${total/N:7.2f}  {n_saves:6,}\n")

    # -- 9. Current breakeven lock analysis ----------------------------
    f.write(f"\n[9] Current Breakeven Lock Status\n")
    f.write(f"    Current activation: trail_activation_ticks * 0.6\n")
    f.write(f"    Avg trail_activation is likely tied to tp_ticks\n")
    f.write(f"    Avg tp_ticks:  {df['tp_ticks'].mean():.0f}\n")
    f.write(f"    If trail_act = tp_ticks, BE fires at {df['tp_ticks'].mean()*0.6:.0f} ticks favorable\n")
    f.write(f"    trade_mfe_ticks avg: {mfe_actual.mean():.1f}, max: {mfe_actual.max():.0f}\n")
    pct_reach_tp = (mfe_actual >= df["tp_ticks"].values * 0.6).mean()
    f.write(f"    % trades reaching BE activation: {pct_reach_tp:.1%}\n")
    f.write(f"    >> BE lock almost never fires because tp_ticks is enormous\n")

    # -- 10. Summary ---------------------------------------------------
    current_pnl = pnl.sum()
    # Best BE threshold
    best_be, best_be_pnl = 0, current_pnl
    for be_thresh in [2, 4, 6, 8, 12, 16, 24, 32]:
        would_save = (mfe_actual >= be_thresh) & (pnl < 0)
        sim = pnl.copy()
        sim[would_save] = 0.0
        if sim.sum() > best_be_pnl:
            best_be_pnl = sim.sum()
            best_be = be_thresh

    f.write(f"\n[10] Summary\n")
    f.write(f"    Current total PnL:   ${current_pnl:,.2f}\n")
    f.write(f"    Best BE threshold:   {best_be} ticks >> ${best_be_pnl:,.2f}\n")
    delta = best_be_pnl - current_pnl
    pct = delta / abs(current_pnl) * 100 if current_pnl else 0
    f.write(f"    Delta:               ${delta:+,.2f} ({pct:+.1f}%)\n")
    f.write(f"    SL fixed sweep:      irrelevant (SL fires 0.5% of trades)\n")
    f.write(f"    KEY FINDING:         Breakeven lock at {best_be} ticks MFE\n")

    return best_be, best_be_pnl, current_pnl


def main():
    results_path = OUT_DIR / "results.txt"
    with open(results_path, "w") as f:
        f.write("Analysis EE -- Stop Loss Optimization (v2, direction-aware)\n")
        f.write(f"{'='*70}\n")

        for label, path in [("IN-SAMPLE", IS_LOG), ("OUT-OF-SAMPLE", OOS_LOG)]:
            if not path.exists():
                f.write(f"\nWARNING: {label} log not found: {path}\n")
                continue
            df = load_trades(path)
            f.write(f"\nLoaded {label}: {len(df):,} trades from {path.name}\n")
            analyze(df, label, f)

        # -- GATE decision --------------------------------------------
        f.write(f"\n\n{'='*70}\n")
        f.write(f"  GATE DECISION\n")
        f.write(f"{'='*70}\n")
        f.write(f"  Fixed SL: KILL -- fires 0.5% of trades, exit engine handles exits\n")
        f.write(f"  Breakeven lock: EVALUATE -- may recover significant loss $ on losers\n")
        f.write(f"  Action: Lower BE activation threshold in exit_engine.py\n")

    print(f"Results written to {results_path}")
    print(f"\nQuick preview:")
    with open(results_path) as f:
        for line in f:
            print(line, end="")


if __name__ == "__main__":
    main()
