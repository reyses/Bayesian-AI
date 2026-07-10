import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Dict, Tuple

import sys
# Ensure we can import core_v2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from core_v2.FPS.forward_pass_system import ForwardPassSystem, BarState

class BayesianHarness:
    def __init__(self, concept_name: str, article_citation: str, registered_response: str, horizon_bars: int = 30, event_mode: str = 'directional'):
        self.concept_name = concept_name
        self.article_citation = article_citation
        self.registered_response = registered_response
        self.horizon_bars = horizon_bars
        self.event_mode = event_mode
        
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        self.atlas_root = os.path.join(self.base_dir, 'DATA/ATLAS')
        self.features_root = os.path.join(self.base_dir, 'DATA/ATLAS/FEATURES_5s_v2')
        self.labels_csv = os.path.join(self.base_dir, 'DATA/ATLAS/regime_labels_2d.csv')
        self.reports_dir = os.path.join(self.base_dir, 'research', 'nt8_catalog', 'reports')
        self.assets_dir = os.path.join(self.reports_dir, 'assets')
        os.makedirs(self.assets_dir, exist_ok=True)

    def process_day(self, args):
        day, concept_class = args
        try:
            fps = ForwardPassSystem(
                day=day, 
                atlas_root=self.atlas_root, 
                features_root=self.features_root, 
                labels_csv=self.labels_csv, 
                build_v2_dict=False
            )
        except FileNotFoundError:
            return None

        concept = concept_class()
        
        records = []
        for state in fps:
            if not state.is_1m_close:
                continue
                
            event_type = concept.eval_state(state) # +1 for Bullish Event, -1 for Bearish, 0 for None
            
            # Need a sigma estimate for the bar to scale the response.
            # We can use the Daily ATR or a rolling standard deviation. 
            # FPS provides state.ohlcv which has history.
            
            records.append({
                'timestamp': state.timestamp,
                'price': state.price,
                'event': event_type
            })
            
        if not records:
            return None
            
        df = pd.DataFrame(records)
        df['fwd_max'] = df['price'].shift(-self.horizon_bars).rolling(self.horizon_bars, min_periods=1).max()
        df['fwd_min'] = df['price'].shift(-self.horizon_bars).rolling(self.horizon_bars, min_periods=1).min()
        
        # We need exact forward indexing to compute first-touch.
        # To keep it fast, we can approximate:
        # Actually, for true first touch, we need a loop or vectorized cummax/cummin.
        # Let's extract events and do the precise calculation just for the event indices.
        
        events = df[df['event'] != 0].copy()
        event_results = []
        
        prices = df['price'].values
        
        # For null controls, we take a random matched sample of non-events.
        null_indices = np.random.choice(df[df['event'] == 0].index, size=min(len(events)*3, len(df[df['event'] == 0])), replace=False)
        
        for idx in list(events.index) + list(null_indices):
            is_event = idx in events.index
            event_val = df.loc[idx, 'event'] if is_event else np.random.choice([-1, 1]) # Phantom direction
            
            if idx + self.horizon_bars >= len(prices):
                continue
                
            path = prices[idx+1 : idx+1+self.horizon_bars]
            p0 = prices[idx]
            
            # Estimate local sigma (e.g. standard deviation of last 30 bars differences)
            if idx >= 30:
                sigma = np.std(np.diff(prices[idx-30:idx+1]))
            else:
                sigma = np.std(np.diff(prices[:idx+1])) if idx > 2 else 1.0
                
            if sigma == 0 or np.isnan(sigma):
                sigma = 1.0 # fallback
                
            # Response logic: First touch bounce
            # Target = 2*sigma, Stop = 2*sigma
            k = 2.0
            target_price = p0 + (k * sigma * event_val)
            stop_price = p0 - (k * sigma * event_val)
            
            hit_target = False
            hit_stop = False
            magnitude = 0.0
            
            for p in path:
                if self.event_mode == 'volatility':
                    # Direction-free breakout
                    if p >= target_price or p <= stop_price: # note target_price is p0 + k*sigma for event_val=1
                        hit_target = True
                        magnitude = max(abs(np.max(path) - p0), abs(np.min(path) - p0)) / sigma
                        break
                else:
                    if event_val > 0:
                        if p >= target_price:
                            hit_target = True
                            magnitude = (np.max(path) - p0) / sigma
                            break
                        elif p <= stop_price:
                            hit_stop = True
                            magnitude = (np.min(path) - p0) / sigma
                            break
                    else:
                        if p <= target_price:
                            hit_target = True
                            magnitude = (p0 - np.min(path)) / sigma
                            break
                        elif p >= stop_price:
                            hit_stop = True
                            magnitude = (p0 - np.max(path)) / sigma
                            break
            
            # If neither hit, it's a timeout.
            if not hit_target and not hit_stop:
                if self.event_mode == 'volatility':
                    magnitude = max(abs(path[-1] - p0), 0) / sigma
                    hit_target = False
                else:
                    magnitude = ((path[-1] - p0) * event_val) / sigma
                    hit_target = magnitude > 0
                
            event_results.append({
                'day': day,
                'is_event': is_event,
                'event_val': event_val,
                'hit_target': int(hit_target),
                'magnitude': magnitude
            })
            
        return event_results

    def run_sweep(self, concept_class, years: List[str] = ['2024', '2025']):
        l0_dir = os.path.join(self.features_root, 'L0')
        all_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
        days = [os.path.basename(f).replace('.parquet', '') for f in all_files]
        days = [d for d in days if any(d.startswith(y) for y in years)]
        
        print(f"[{self.concept_name}] Starting Bayesian sweep over {len(days)} days in {years}...")
        
        args = [(day, concept_class) for day in days]
        
        all_results = []
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
            for res in executor.map(self.process_day, args):
                if res is not None:
                    all_results.extend(res)
                    
        df = pd.DataFrame(all_results)
        
        if len(df) == 0:
            print("No events found.")
            return
            
        events_df = df[df['is_event'] == True]
        null_df = df[df['is_event'] == False]
        
        n_events = len(events_df)
        p_resp_event = events_df['hit_target'].mean() if n_events > 0 else 0.0
        p_resp_null = null_df['hit_target'].mean() if len(null_df) > 0 else 0.0
        
        # Bimodality check via crude histogram peak counting or median vs mean
        magnitudes = events_df['magnitude'].dropna().values
        mode_mag = np.round(pd.Series(np.round(magnitudes, 1)).mode().values[0], 2) if len(magnitudes) > 0 else 0.0
        median_mag = np.median(magnitudes) if len(magnitudes) > 0 else 0.0
        tail_mag = np.percentile(magnitudes, 90) if len(magnitudes) > 0 else 0.0
        
        is_bimodal = abs(mode_mag - median_mag) > 1.0 # Crude heuristic
        
        # Day-block CI for P(resp)
        B = 1000
        days_list = events_df['day'].unique()
        boot_ps = []
        for _ in range(B):
            idx = np.random.choice(days_list, size=len(days_list), replace=True)
            boot_df = events_df[events_df['day'].isin(idx)]
            boot_ps.append(boot_df['hit_target'].mean())
            
        ci_lower = np.percentile(boot_ps, 2.5) if boot_ps else 0.0
        ci_upper = np.percentile(boot_ps, 97.5) if boot_ps else 0.0
        
        # Plot Magnitude
        plt.figure(figsize=(8,5))
        plt.hist(magnitudes, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(mode_mag, color='red', linestyle='dashed', linewidth=2, label=f'Mode: {mode_mag}')
        plt.title(f"{self.concept_name} - Magnitude Distribution ($\sigma$-scaled)")
        plt.xlabel("Magnitude ($\sigma$)")
        plt.ylabel("Frequency")
        plt.legend()
        fig_path = os.path.join(self.assets_dir, f"{self.concept_name.replace(' ', '_')}_mag.png")
        plt.savefig(fig_path)
        plt.close()
        
        gap = p_resp_event - p_resp_null
        verdict = "NOISE"
        if gap > 0.10 and ci_lower > p_resp_null:
            verdict = "REAL"
        elif gap > 0.05 and ci_lower > p_resp_null:
            verdict = "CONDITIONAL"
            
        report_md = f"""# Concept Report: {self.concept_name}

## 1. Source & Definition
- **Citation:** `{self.article_citation}`
- **Registered Response:** {self.registered_response}
- **Event Definition:** {self.concept_name}

## 2. Event Probabilities (vs Null)
- **N Events:** {n_events}
- **Phantom Null Base Rate P(Resp):** {p_resp_null:.4f}
- **Event P(Resp):** {p_resp_event:.4f}
- **Bayesian Edge (Delta):** +{(gap * 100):.2f} pp
- **95% Day-Block CI for P(Resp):** [{ci_lower:.4f}, {ci_upper:.4f}]

## 3. Magnitude Distribution ($\sigma$-scaled)
- **Mode (Bulk):** {mode_mag} $\sigma$
- **Median:** {median_mag:.2f} $\sigma$
- **90th Percentile Tail:** {tail_mag:.2f} $\sigma$
- **Bimodal Flag:** {is_bimodal}

![Magnitude Distribution](assets/{self.concept_name.replace(' ', '_')}_mag.png)

## 4. Verdict
**{verdict}**
"""
        
        report_path = os.path.join(self.reports_dir, f"AG_cat_{self.concept_name.replace(' ', '_')}.md")
        with open(report_path, 'w') as f:
            f.write(report_md)
            
        print(f"[{self.concept_name}] Report written to {report_path}")
        
        # Update Master Index
        index_path = os.path.join(self.reports_dir, 'AG_cat_00_INDEX.md')
        with open(index_path, 'a') as f:
            f.write(f"| {self.concept_name} | VWAP Touch | {self.registered_response} | {n_events} | {p_resp_event:.2f} (vs {p_resp_null:.2f}) | {mode_mag} | {tail_mag:.2f} | {is_bimodal} | **{verdict}** |\n")

