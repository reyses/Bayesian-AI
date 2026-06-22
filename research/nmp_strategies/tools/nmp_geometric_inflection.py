"""Session-Anchored Geometric Inflection simulation.
Direction: 5m mean slope (L2_5m_price_velocity_9)
Location: Pure geometric points (Local Coordinate System / Anchors)

Rules:
1. Start session: Drop anchor. Track local_max and local_min.
2. LONG entry: If slope > SLOPE_THRESH (Vector proven) AND price drops DIP_PTS from local_max (Find best price).
3. LONG exit: If slope drops by CURVE_THRESH from its peak since entry (Curving) OR drops below 0. 
   -> Exit and drop NEW ANCHOR.
4. SHORT entry: If slope < -SLOPE_THRESH AND price rises DIP_PTS from local_min.
5. SHORT exit: If slope rises by CURVE_THRESH from its lowest point since entry OR rises above 0.
   -> Exit and drop NEW ANCHOR.
"""
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem

SLOPE_THRESH = 0.5
DIP_PTS = 5.0
CURVE_THRESH = 0.5

ATLAS = 'DATA/ATLAS'
FEAT = f'{ATLAS}/FEATURES_5s_v2'
LABELS = f'{ATLAS}/regime_labels_2d.csv'

USD_PER_PT = 20.0
COMMISSION_PER_SIDE = 2.05
SLIPPAGE_PTS_PER_SIDE = 0.25
COST_PER_TRADE_USD = (COMMISSION_PER_SIDE * 2) + (SLIPPAGE_PTS_PER_SIDE * 2 * USD_PER_PT)


def dayblock_ci(per_day):
    nd = len(per_day)
    RNG = np.random.RandomState(42)
    boots = [per_day[RNG.randint(0, nd, nd)].mean() for _ in range(4000)]
    return float(per_day.mean()), tuple(np.percentile(boots, [2.5, 97.5]))


def main():
    l0_dir = os.path.join(FEAT, 'L0')
    files = sorted(glob.glob(os.path.join(l0_dir, '2024_*.parquet')))
    days = [os.path.basename(f).replace('.parquet', '') for f in files]

    if not days:
        print("No 2024 data found!")
        return

    fps = MultiDayForwardPassSystem(
        atlas_root=ATLAS,
        features_root=FEAT,
        labels_csv=LABELS,
        days=days
    )

    trades = []
    holds = []
    
    pos = 0
    ei = None
    edir = None
    eprice = 0.0
    ets = 0
    
    # State tracking
    current_day = None
    anchor_px = None
    local_max = None
    local_min = None
    
    # Track the peak/valley of the slope to detect curving
    extreme_slope = 0.0
    
    for state in fps:
        # Reset anchor on new day
        if state.day != current_day:
            current_day = state.day
            anchor_px = None
            pos = 0 # force close overnight
        
        if not state.is_1m_close:
            continue
            
        slope_5m = state.v2.get('L2_5m_price_velocity_9', np.nan)
        if np.isnan(slope_5m):
            continue
            
        ts = state.timestamp
        px = state.price
        
        # 1. Coordinate System Initialization
        if anchor_px is None:
            anchor_px = px
            local_max = px
            local_min = px
            continue
            
        # Update local extremes
        local_max = max(local_max, px)
        local_min = min(local_min, px)
        
        if pos == 0:
            # Look for LONG entry
            if slope_5m > SLOPE_THRESH:
                # Vector is UP. Find best price.
                if (local_max - px) >= DIP_PTS:
                    pos = 1
                    edir = 'LONG'
                    eprice = px
                    ets = ts
                    extreme_slope = slope_5m
            # Look for SHORT entry
            elif slope_5m < -SLOPE_THRESH:
                # Vector is DOWN. Find best price.
                if (px - local_min) >= DIP_PTS:
                    pos = -1
                    edir = 'SHORT'
                    eprice = px
                    ets = ts
                    extreme_slope = slope_5m
        else:
            exit_reason = None
            if pos == 1:
                extreme_slope = max(extreme_slope, slope_5m)
                # Exit if slope curves over OR completely breaks
                if (extreme_slope - slope_5m >= CURVE_THRESH) or slope_5m <= 0:
                    exit_reason = 'curve_deceleration'
                    
            elif pos == -1:
                extreme_slope = min(extreme_slope, slope_5m)
                # Exit if slope curves up OR completely breaks
                if (slope_5m - extreme_slope >= CURVE_THRESH) or slope_5m >= 0:
                    exit_reason = 'curve_deceleration'
                    
            if exit_reason:
                pts = (px - eprice) if pos == 1 else (eprice - px)
                usd_gross = pts * USD_PER_PT
                usd_net = usd_gross - COST_PER_TRADE_USD
                trades.append((state.day, int(ets), edir, float(eprice), int(ts), float(px),
                               float(pts), float(usd_gross), float(usd_net), exit_reason))
                holds.append((ts - ets) / 60.0)
                pos = 0
                
                # 2. Drop NEW Localized Anchor at exit
                anchor_px = px
                local_max = px
                local_min = px

    # Close out final trade if any
    if pos != 0:
        pts = (px - eprice) if pos == 1 else (eprice - px)
        usd_gross = pts * USD_PER_PT
        usd_net = usd_gross - COST_PER_TRADE_USD
        trades.append((state.day, int(ets), edir, float(eprice), int(ts), float(px),
                       float(pts), float(usd_gross), float(usd_net), 'eod'))
        holds.append((ts - ets) / 60.0)

    tr = pd.DataFrame(trades, columns=['day', 'entry_ts', 'leg_dir', 'entry_price', 'exit_ts',
                                       'exit_price', 'pnl_pts', 'gross_usd', 'net_usd', 'exit_reason'])
    
    out_csv = 'reports/findings/nmp_geometric_inflection_2024.csv'
    os.makedirs('reports/findings', exist_ok=True)
    tr.to_csv(out_csv, index=False)

    if len(tr) == 0:
        print("No trades generated.")
        return

    p_net = tr['net_usd'].to_numpy()
    p_gross = tr['gross_usd'].to_numpy()
    
    win = p_net[p_net > 0].sum()
    loss = abs(p_net[p_net < 0].sum())
    pf = win / loss if loss > 0 else float('inf')
    
    per_day = tr.groupby('day')['net_usd'].sum().reindex(days).fillna(0).to_numpy()
    m, ci = dayblock_ci(per_day)

    L = []
    L.append("# NMP Geometric Inflection (2024)\n")
    L.append(f"Direction: 5m slope (Threshold: {SLOPE_THRESH})")
    L.append(f"Entry: Point Pullback (Dip/Rally >= {DIP_PTS} pts from local extreme)")
    L.append(f"Exit: Curve Deceleration (Slope drops {CURVE_THRESH} from its extreme)")
    L.append(f"Costs: {COST_PER_TRADE_USD:.2f} USD per trade round-trip included.\n")
    
    L.append("| metric | **Geometric Inflection Net** |")
    L.append("|---|---|")
    L.append(f"| trades | {len(tr)} |")
    L.append(f"| trades/day | {len(tr)/len(days):.1f} |")
    L.append(f"| Net $/trade mean | {p_net.mean():+.2f} |")
    L.append(f"| Gross $/trade mean | {p_gross.mean():+.2f} |")
    L.append(f"| PF-based Trade WR | {pf-1:+.3f} (PF {pf:.2f}) |")
    L.append(f"| Net $/day mean | {m:+.0f} |")
    L.append(f"| Net $/day 95% day-block CI | [{ci[0]:+.0f}, {ci[1]:+.0f}] |")
    L.append(f"| winning days | {(per_day>0).sum()}/{len(days)} |")
    L.append(f"| avg hold (min) | {np.mean(holds) if len(holds) else 0:.1f} |")
    L.append(f"\nVerdict: NET PROFIT {p_net.sum():+.0f} USD. Mean $/day {m:+.0f} USD.")
    
    rep = '\n'.join(L)
    open('reports/findings/nmp_geometric_inflection_2024.md', 'w').write(rep)
    print(rep)
    print(f"\n[trades -> {out_csv}]")


if __name__ == '__main__':
    main()
