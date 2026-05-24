"""Quick analyzer for NT8 native backtest trade-export CSVs.

Different schema from the strategy's own CSV ledger. Columns include
Profit (formatted as $X.XX or ($X.XX) for losses), MAE/MFE/ETD in dollars.
"""
import csv
import sys


def parse_money(s):
    s = s.replace("$", "").replace(",", "").strip()
    if s.startswith("(") and s.endswith(")"):
        return -float(s[1:-1])
    return float(s) if s else 0.0


def main(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    reasons = {}
    losses_5pt = 0   # MAE ~$10 (= 5pt), MFE <$3, pnl negative
    losses_lt15 = 0  # any loss between -$15 and 0 with low MFE
    total_pnl = 0
    pnls = []
    bars_dist = {}  # bucket by held-bar count
    for r in rows:
        pnl = parse_money(r["Profit"])
        pnls.append(pnl)
        mfe = parse_money(r["MFE"])
        mae = parse_money(r["MAE"])
        reason = r["Exit name"]
        reasons[reason] = reasons.get(reason, 0) + 1
        total_pnl += pnl
        if 9 < mae < 11 and mfe < 3 and pnl < 0:
            losses_5pt += 1
        if -15 < pnl < 0 and mfe < 3:
            losses_lt15 += 1
        bars = int(r["Bars"]) if r["Bars"].strip() else 0
        bucket = "1-5" if bars <= 5 else ("6-30" if bars <= 30 else ("31-100" if bars <= 100 else ">100"))
        bars_dist[bucket] = bars_dist.get(bucket, 0) + 1

    print(f"N trades:        {len(rows)}")
    print(f"Total PnL:       ${total_pnl:.0f}")
    print(f"Mean PnL/trade:  ${total_pnl/len(rows):.2f}")
    print(f"Median PnL:      ${sorted(pnls)[len(pnls)//2]:.2f}")
    print(f"Min/Max PnL:     ${min(pnls):.2f} / ${max(pnls):.2f}")
    print()
    print("Exit reasons:")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k:30s} {v:6d}  ({100*v/len(rows):5.1f}%)")
    print()
    print("Held-bar distribution:")
    for k in ["1-5", "6-30", "31-100", ">100"]:
        v = bars_dist.get(k, 0)
        print(f"  {k:10s} {v:6d}  ({100*v/len(rows):5.1f}%)")
    print()
    print(f"Stopped at ~5pt loss with NO favorable move (MAE~$10, MFE<$3): {losses_5pt}  ({100*losses_5pt/len(rows):.1f}%)")
    print(f"Any loss between -$15 and 0 with MFE<$3:                       {losses_lt15}  ({100*losses_lt15/len(rows):.1f}%)")


if __name__ == "__main__":
    main(sys.argv[1])
