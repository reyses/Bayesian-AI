import os
import glob
import sys
import numpy as np
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Ensure we can import the harness and concepts
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ag_cat_01_vwap_pullback import VWAPPullbackConcept
from ag_cat_03_apz_touches import APZTouchesConcept
from ag_cat_04_squeeze import SqueezeConcept
from ag_cat_05_candle_shapes import CandleShapeConcept
from ag_cat_06_ma_crossover import MACrossoverConcept

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from core_v2.FPS.forward_pass_system import ForwardPassSystem

def process_joint_day(day):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    atlas_root = os.path.join(base_dir, 'DATA/ATLAS')
    features_root = os.path.join(base_dir, 'DATA/ATLAS/FEATURES_5s_v2')
    labels_csv = os.path.join(base_dir, 'DATA/ATLAS/regime_labels_2d.csv')
    
    try:
        fps = ForwardPassSystem(day=day, atlas_root=atlas_root, features_root=features_root, labels_csv=labels_csv, build_v2_dict=False)
    except FileNotFoundError:
        return None

    c_vwap = VWAPPullbackConcept()
    c_apz = APZTouchesConcept()
    c_sqz = SqueezeConcept()
    c_can = CandleShapeConcept()
    c_ma = MACrossoverConcept()
    
    records = []
    prices = []
    
    idx = 0
    for state in fps:
        prices.append(state.price)
        if not state.is_1m_close:
            idx += 1
            continue
            
        v_vwap = c_vwap.eval_state(state)
        v_apz = c_apz.eval_state(state)
        v_sqz = c_sqz.eval_state(state)
        v_can = c_can.eval_state(state)
        v_ma = c_ma.eval_state(state)
        
        # Local sigma
        if idx >= 30:
            sigma = np.std(np.diff(prices[idx-30:idx+1]))
        else:
            sigma = np.std(np.diff(prices[:idx+1])) if idx > 2 else 1.0
            
        if sigma == 0 or np.isnan(sigma):
            sigma = 1.0
            
        context = {
            'vwap': v_vwap,
            'apz': v_apz,
            'sqz': v_sqz,
            'can': v_can,
            'ma': v_ma
        }
        
        # Create an event row for EACH trigger
        if v_vwap != 0: records.append({'day': day, 'idx': idx, 'trigger': 'vwap', 'event_val': v_vwap, 'sigma': sigma, 'mode': 'directional', **context})
        if v_apz != 0:  records.append({'day': day, 'idx': idx, 'trigger': 'apz', 'event_val': v_apz, 'sigma': sigma, 'mode': 'directional', **context})
        if v_sqz != 0:  records.append({'day': day, 'idx': idx, 'trigger': 'sqz', 'event_val': v_sqz, 'sigma': sigma, 'mode': 'volatility', **context})
        if v_can != 0:  records.append({'day': day, 'idx': idx, 'trigger': 'can', 'event_val': v_can, 'sigma': sigma, 'mode': 'directional', **context})
        if v_ma != 0:   records.append({'day': day, 'idx': idx, 'trigger': 'ma', 'event_val': v_ma, 'sigma': sigma, 'mode': 'directional', **context})
            
        idx += 1
            
    if not records:
        return None
        
    prices_array = np.array(prices)
    horizon_bars = 60
    k = 2.0
    
    event_results = []
    
    for r in records:
        idx = r['idx']
        if idx + horizon_bars >= len(prices_array):
            continue
            
        path = prices_array[idx+1 : idx+1+horizon_bars]
        p0 = prices_array[idx]
        sigma = r['sigma']
        event_val = r['event_val']
        mode = r['mode']
        
        target_price = p0 + (k * sigma * event_val)
        stop_price = p0 - (k * sigma * event_val)
        
        hit_target = False
        hit_stop = False
        
        for p in path:
            if mode == 'volatility':
                if p >= target_price or p <= stop_price: 
                    hit_target = True
                    break
            else:
                if event_val > 0:
                    if p >= target_price:
                        hit_target = True
                        break
                    elif p <= stop_price:
                        break
                else:
                    if p <= target_price:
                        hit_target = True
                        break
                    elif p >= stop_price:
                        break
                        
        if not hit_target and not hit_stop:
            if mode == 'volatility':
                hit_target = False
            else:
                magnitude = ((path[-1] - p0) * event_val) / sigma
                hit_target = magnitude > 0
                
        r['claim_was_true'] = int(hit_target)
        event_results.append(r)
        
    return event_results

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    features_root = os.path.join(base_dir, 'DATA/ATLAS/FEATURES_5s_v2')
    l0_dir = os.path.join(features_root, 'L0')
    all_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    days = [os.path.basename(f).replace('.parquet', '') for f in all_files]
    days = [d for d in days if d.startswith('2024')]
    
    print(f"[Mechanical EDA] Scanning {len(days)} days in 2024...")
    
    all_results = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
        for res in executor.map(process_joint_day, days):
            if res is not None:
                all_results.extend(res)
                
    df = pd.DataFrame(all_results)
    
    report_lines = []
    report_lines.append("# Joint Confluence: Mechanical EDA")
    report_lines.append("This report explicitly counts conditional intersections of indicators to evaluate confluence without black-box ML models.")
    report_lines.append("")
    
    signals = ['vwap', 'apz', 'sqz', 'can', 'ma']
    
    report_lines.append("## 1. Confluence Impact Table")
    report_lines.append("| Base Signal | Confluence Signal | Total Base Events | Base Win Rate | Confluence Events (N) | Confluence Win Rate | Lift (pp) |")
    report_lines.append("|---|---|---|---|---|---|---|")
    
    for base in signals:
        df_base = df[df['trigger'] == base]
        if len(df_base) == 0: continue
            
        base_wr = df_base['claim_was_true'].mean()
        base_n = len(df_base)
        
        for conf in signals:
            if conf == base:
                continue
                
            # Filter where the confluence signal also fired on the same bar
            df_conf = df_base[df_base[conf] != 0]
            if len(df_conf) == 0:
                continue
                
            conf_wr = df_conf['claim_was_true'].mean()
            conf_n = len(df_conf)
            lift = (conf_wr - base_wr) * 100
            
            report_lines.append(f"| {base.upper()} | + {conf.upper()} | {base_n} | {base_wr:.4f} | {conf_n} | {conf_wr:.4f} | {lift:+.2f} pp |")
            
    report_path = os.path.join(base_dir, 'research', 'nt8_catalog', 'reports', 'AG_Joint_EDA.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"[Mechanical EDA] Complete. Saved to {report_path}")
