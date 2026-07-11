import os
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Callable
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from core_v2.FPS.forward_pass_system import ForwardPassSystem, BarState

# Cost of trading
TICK_SIZE = 0.25
TICK_VALUE = 0.50  # MNQ
ROUND_TRIP_TICKS = 4
SLIPPAGE_USD = ROUND_TRIP_TICKS * TICK_VALUE

class ConceptHarness:
    """
    Base class for any concept test.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Called at the start of each day to reset any rolling window state."""
        pass
        
    def eval_state(self, state: BarState) -> float:
        """
        Called every 5s bar.
        Must return a signal between -1.0 (Short) and 1.0 (Long).
        0.0 means no position.
        """
        raise NotImplementedError()


def process_day(args):
    day, concept_class, atlas_root, features_root, labels_csv = args
    try:
        fps = ForwardPassSystem(
            day=day, 
            atlas_root=atlas_root, 
            features_root=features_root, 
            labels_csv=labels_csv, 
            build_v2_dict=False  # Faster execution, rely on ohlcv dictionaries
        )
    except FileNotFoundError:
        return None

    concept = concept_class()
    concept.reset()
    
    records = []
    
    for state in fps:
        # Only process on 1-minute close to keep the signal cardinality manageable
        if not state.is_1m_close:
            continue
            
        signal = concept.eval_state(state)
        
        records.append({
            'timestamp': state.timestamp,
            'price': state.price,
            'signal': signal
        })
        
    if len(records) == 0:
        return None
        
    df = pd.DataFrame(records)
    
    # 1. Forward Return Calculation (15m forward)
    # 15m is 15 bars because we only kept 1m closes
    df['fwd_price_15m'] = df['price'].shift(-15)
    df['fwd_ret_15m'] = df['fwd_price_15m'] - df['price']
    
    # 2. Daily Economics (Vectorized backtest)
    df['position'] = df['signal'].shift(1).fillna(0)
    df['trade_ret'] = df['position'] * (df['price'].diff())
    
    # Count trades (when position changes)
    df['trade_entry'] = (df['position'] != df['position'].shift(1)) & (df['position'] != 0)
    num_trades = df['trade_entry'].sum()
    
    # Compute gross PnL
    # Assuming trading 1 MNQ contract, $2 per point = $0.50 per tick. 
    # Price difference is in points. 1 point = 4 ticks = $2.00
    gross_pnl_pts = df['trade_ret'].sum()
    gross_pnl_usd = gross_pnl_pts * 2.0
    
    # Apply costs
    net_pnl_usd = gross_pnl_usd - (num_trades * SLIPPAGE_USD)
    
    # 3. Null-Control Stats
    # Expected return given signal vs overall expected return (drift)
    long_mask = df['signal'] > 0.5
    short_mask = df['signal'] < -0.5
    
    long_ret = df.loc[long_mask, 'fwd_ret_15m'].mean() if long_mask.sum() > 0 else 0.0
    short_ret = df.loc[short_mask, 'fwd_ret_15m'].mean() if short_mask.sum() > 0 else 0.0
    
    # Baseline expected return for matching time of day (naive mean)
    baseline_ret = df['fwd_ret_15m'].mean()
    
    return {
        'day': day,
        'net_pnl_usd': net_pnl_usd,
        'gross_pnl_usd': gross_pnl_usd,
        'num_trades': num_trades,
        'long_ret': long_ret,
        'short_ret': short_ret,
        'baseline_ret': baseline_ret,
        'signal_count': long_mask.sum() + short_mask.sum()
    }


def run_sweep(concept_class, concept_name: str, years: List[str] = ['2024', '2025']):
    """
    Runs the concept sweep in parallel across all days in the specified years.
    Outputs a Markdown report.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    atlas_root = os.path.join(base_dir, 'DATA/ATLAS')
    features_root = os.path.join(base_dir, 'DATA/ATLAS/FEATURES_5s_v2')
    labels_csv = os.path.join(base_dir, 'DATA/ATLAS/regime_labels_2d.csv')
    
    l0_dir = os.path.join(features_root, 'L0')
    all_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    days = [os.path.basename(f).replace('.parquet', '') for f in all_files]
    
    # Filter by years
    days = [d for d in days if any(d.startswith(y) for y in years)]
    
    print(f"[{concept_name}] Starting causal sweep over {len(days)} days in {years}...")
    
    args = [(day, concept_class, atlas_root, features_root, labels_csv) for day in days]
    
    results = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
        for res in executor.map(process_day, args):
            if res is not None:
                results.append(res)
                
    if not results:
        print(f"[{concept_name}] No results generated. Check data paths.")
        return
        
    df = pd.DataFrame(results)
    
    # --- Block Bootstrap CI (95%) for Net PnL ---
    B = 1000
    means = np.zeros(B)
    n_days = len(df)
    pnl_array = df['net_pnl_usd'].values
    for i in range(B):
        idx = np.random.randint(0, n_days, n_days)
        means[i] = pnl_array[idx].mean()
        
    ci_lower = np.percentile(means, 2.5)
    ci_upper = np.percentile(means, 97.5)
    mean_pnl = np.mean(means)
    
    # --- Null Control Stats ---
    # Real edge is gap between signal return and baseline return
    # Gap = |long_ret - baseline| + |-short_ret - baseline|
    avg_long = df['long_ret'].mean()
    avg_short = df['short_ret'].mean()
    avg_base = df['baseline_ret'].mean()
    
    gap_long = avg_long - avg_base
    gap_short = avg_base - avg_short # Short should be negative, so base - short is positive edge
    
    total_gap = gap_long + gap_short
    
    verdict = "NOISE"
    verdict_reason = f"Bootstrap CI [{ci_lower:.2f}, {ci_upper:.2f}] includes 0 or is highly negative."
    
    if ci_lower > 0 and total_gap > 0.10:
        verdict = "REAL"
        verdict_reason = f"Bootstrap CI is strictly positive and gap ({total_gap:.2f}) >= 0.10"
    elif ci_lower > 0 and total_gap > 0.05:
        verdict = "CONDITIONAL"
        verdict_reason = f"Bootstrap CI is strictly positive and gap ({total_gap:.2f}) >= 0.05"
        
    report_md = f"""# Concept Report: {concept_name}

## 1. Definition
- **Concept:** {concept_name}
- **Methodology:** Causal forward-pass (no lookahead). Block-bootstrap over {n_days} days.

## 2. Existence Test (Null Control)
- **Baseline Expected 15m Return:** {avg_base:.4f} pts
- **Long Signal 15m Return:** {avg_long:.4f} pts (Gap: {gap_long:.4f})
- **Short Signal 15m Return:** {avg_short:.4f} pts (Gap: {gap_short:.4f})
- **Combined Edge Gap:** {total_gap:.4f} pts

## 3. Economics Test
- **Average Trades / Day:** {df['num_trades'].mean():.2f}
- **Gross PnL / Day:** ${df['gross_pnl_usd'].mean():.2f}
- **Net PnL / Day (4 ticks round-trip cost):** ${mean_pnl:.2f}
- **95% Bootstrap CI (Net $/day):** [${ci_lower:.2f}, ${ci_upper:.2f}]

## 4. Verdict
**{verdict}**
*Reasoning:* {verdict_reason}
"""
    
    reports_dir = os.path.join(base_dir, 'research', 'leg_clock', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"AG_cat_{concept_name.replace(' ', '_')}.md")
    
    with open(report_path, 'w') as f:
        f.write(report_md)
        
    print(f"[{concept_name}] Report written to {report_path}")
    print(f"Verdict: {verdict} (Mean Net $/day: ${mean_pnl:.2f})")
