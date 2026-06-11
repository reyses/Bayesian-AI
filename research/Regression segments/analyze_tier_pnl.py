import json
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

VALID_TIER_MAX = 9

def analyze_tier_vs_pnl():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    oos_path = os.path.join(repo_root, "oos_trade_data.json")
    
    if not os.path.exists(oos_path):
        print(f"Error: {oos_path} not found.")
        return
        
    with open(oos_path, 'r') as f:
        oos_data = json.load(f)
        
    if 'metadata' not in oos_data:
        print("Error: 'metadata' missing from oos_trade_data.json.")
        print("Please re-run the ML training/eval (train_gpu_research_A.py) so it dumps the entry indices.")
        return
        
    pnls = oos_data['pnls']
    metadata = oos_data['metadata'] # list of [day, entry_bar, exit_bar, agent_dir]
    
    if len(pnls) != len(metadata):
        print(f"Mismatch: {len(pnls)} PnLs vs {len(metadata)} metadata entries.")
        return
        
    # Preload segments for the days present in the trades
    days_in_trades = list(set([m[0] for m in metadata]))
    day_segments = {}
    
    print(f"Trades span {len(days_in_trades)} days. Loading segment artifacts...")
    for day in days_in_trades:
        # Prefer stage 2 (finer chaos resolution), fallback to stage 1
        s2_path = os.path.join(repo_root, "artifacts", f"stage2_segments_{day}.json")
        s1_path = os.path.join(repo_root, "artifacts", f"stage1_segments_{day}.json")
        
        if os.path.exists(s2_path):
            with open(s2_path, 'r') as f:
                day_segments[day] = json.load(f)
        elif os.path.exists(s1_path):
            with open(s1_path, 'r') as f:
                day_segments[day] = json.load(f)
        else:
            print(f"Warning: No segment data found for {day}. Trades on this day will be skipped.")
            day_segments[day] = []
            
    # Map each trade to its tier
    trade_results = []
    
    has_warned_raw_idx = False
    excluded_trades_count = 0
    
    for pnl, meta in zip(pnls, metadata):
        day, entry_bar, exit_bar, agent_dir = meta
        segs = day_segments.get(day, [])
        
        assigned_tier = None
        tier_journey = []
        unclassified_overlap_len = 0
        pct_unclassified = 0.0
        
        if exit_bar == entry_bar:
            for s in segs:
                if not has_warned_raw_idx and 'raw_start_idx' not in s:
                    print(f"Warning: segments for {day} predate raw-index fix — coordinates may be shifted; re-run stage2 for this day")
                    has_warned_raw_idx = True
                    
                s_idx = s.get('raw_start_idx', s['start_idx'])
                e_idx = s.get('raw_end_idx', s['end_idx'])

                if s_idx <= entry_bar < e_idx:
                    raw_tier = s.get('volatility_tier', 99)
                    if isinstance(raw_tier, (int, float)) and raw_tier <= VALID_TIER_MAX:
                        assigned_tier = float(raw_tier)
                        tier_journey.append((int(assigned_tier), 1.0))
                    else:
                        unclassified_overlap_len = 1
                        pct_unclassified = 1.0
                        tier_journey.append(('?', 1.0))
                        assigned_tier = None
                        excluded_trades_count += 1
                    break
        else:
            weighted_tier_sum = 0.0
            overlap_sum = 0
            journey_parts = []
            
            for s in segs:
                if not has_warned_raw_idx and 'raw_start_idx' not in s:
                    print(f"Warning: segments for {day} predate raw-index fix — coordinates may be shifted; re-run stage2 for this day")
                    has_warned_raw_idx = True
                    
                s_idx = s.get('raw_start_idx', s['start_idx'])
                e_idx = s.get('raw_end_idx', s['end_idx'])

                overlap_start = max(entry_bar, s_idx)
                overlap_end = min(exit_bar, e_idx)
                
                if overlap_start < overlap_end:
                    overlap_len = overlap_end - overlap_start
                    raw_tier = s.get('volatility_tier', 99)
                    
                    if isinstance(raw_tier, (int, float)) and raw_tier <= VALID_TIER_MAX:
                        overlap_sum += overlap_len
                        tier_val = float(raw_tier)
                        weighted_tier_sum += overlap_len * tier_val
                        journey_parts.append((int(tier_val), overlap_len))
                    else:
                        unclassified_overlap_len += overlap_len
                        journey_parts.append(('?', overlap_len))
                        
            total_trade_len = exit_bar - entry_bar
            if total_trade_len > 0:
                pct_unclassified = unclassified_overlap_len / total_trade_len
                
            if pct_unclassified > 0.5:
                assigned_tier = None
                excluded_trades_count += 1
            elif overlap_sum > 0:
                assigned_tier = weighted_tier_sum / overlap_sum
                
            total_covered = overlap_sum + unclassified_overlap_len
            if total_covered > 0:
                tier_journey = [(t, length / total_covered) for t, length in journey_parts]
                
        journey_str = " -> ".join([f"T{t}({pct*100:.0f}%)" if t != '?' else f"T?({(pct)*100:.0f}%)" for t, pct in tier_journey])
        
        trade_results.append({
            'day': day,
            'entry_bar': entry_bar,
            'exit_bar': exit_bar,
            'tier': assigned_tier,
            'pct_unclassified': pct_unclassified,
            'pnl': pnl,
            'is_win': 1 if pnl > 0 else 0,
            'tier_journey': journey_str
        })
        
    if excluded_trades_count > 0:
        print(f"Note: Excluded {excluded_trades_count} trades due to >50% unclassified overlap.")
            
    if not trade_results:
        print("Could not map any trades to segment tiers. Are the artifacts complete?")
        return
        
    df = pd.DataFrame(trade_results)
    
    # Save the detailed trade-by-trade journey to a CSV for manual inspection
    csv_out = os.path.join(repo_root, "trade_tier_journeys.csv")
    df.to_csv(csv_out, index=False)
    print(f"\n[+] Detailed chronological tier journeys saved to: {csv_out}")
    
    print("\n" + "="*50)
    print("📊 TIER vs PNL DIAGNOSTIC REPORT")
    print("="*50)
    
    # Filter out unclassified trades before correlations and summary
    valid_df = df.dropna(subset=['tier']).copy()
    
    # Aggregate by Tier (round weighted tiers to nearest integer for the grouping summary)
    valid_df['rounded_tier'] = valid_df['tier'].round().astype(int)
    summary = []
    for tier in sorted(valid_df['rounded_tier'].unique()):
        tier_df = valid_df[valid_df['rounded_tier'] == tier]
        n_trades = len(tier_df)
        win_rate = tier_df['is_win'].mean() * 100
        avg_pnl = tier_df['pnl'].mean()
        net_pnl = tier_df['pnl'].sum()
        
        summary.append({
            'Rounded Tier': tier,
            'Trades': n_trades,
            'Win Rate %': win_rate,
            'Avg PnL': avg_pnl,
            'Net PnL': net_pnl
        })
        
    if summary:
        summary_df = pd.DataFrame(summary).set_index('Rounded Tier')
        print(summary_df.to_string(float_format=lambda x: f"{x:.2f}"))
    else:
        print("No valid trades remaining after tier exclusion.")
        
    print("\n" + "="*50)
    print("📈 CORRELATION METRICS")
    print("="*50)
    
    if len(valid_df) < 2:
        print("Not enough valid trades for correlation.")
        return
        
    # Calculate Correlation between Tier (x) and PnL (y)
    # A negative correlation means Higher Tier (more chaos) = Lower PnL
    tiers = valid_df['tier'].values
    pnls_array = valid_df['pnl'].values
    
    pearson_corr, p_p = pearsonr(tiers, pnls_array)
    spearman_corr, p_s = spearmanr(tiers, pnls_array)
    
    print(f"Pearson Correlation:  {pearson_corr:+.4f} (p-value: {p_p:.4f})")
    print(f"Spearman Correlation: {spearman_corr:+.4f} (p-value: {p_s:.4f})")
    
    if pearson_corr < -0.1 and p_p < 0.05:
        print("\n✅ DIAGNOSTIC: Statistically significant NEGATIVE correlation found!")
        print("   This confirms the ML model performs worse during high-chaos (high tier) regimes.")
    elif pearson_corr > 0.1 and p_p < 0.05:
        print("\n⚠️ DIAGNOSTIC: Statistically significant POSITIVE correlation found!")
        print("   The ML model actually performs *better* during high-chaos regimes.")
    else:
        print("\nℹ️ DIAGNOSTIC: No strong linear correlation found between regime tier and PnL.")

if __name__ == '__main__':
    analyze_tier_vs_pnl()
