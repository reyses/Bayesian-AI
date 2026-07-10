import os
import glob
import sys
import numpy as np
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
        
        # We need local sigma
        if idx >= 30:
            sigma = np.std(np.diff(prices[idx-30:idx+1]))
        else:
            sigma = np.std(np.diff(prices[:idx+1])) if idx > 2 else 1.0
            
        if sigma == 0 or np.isnan(sigma):
            sigma = 1.0
            
        # The joint state context X
        context = {
            'vwap_state': v_vwap,
            'apz_state': v_apz,
            'sqz_state': v_sqz,
            'can_state': v_can,
            'ma_state': v_ma
        }
        
        # If any triggered, we create an event row for EACH trigger
        if v_vwap != 0:
            records.append({'day': day, 'idx': idx, 'trigger': 'VWAP', 'event_val': v_vwap, 'sigma': sigma, 'mode': 'directional', **context})
        if v_apz != 0:
            records.append({'day': day, 'idx': idx, 'trigger': 'APZ', 'event_val': v_apz, 'sigma': sigma, 'mode': 'directional', **context})
        if v_sqz != 0:
            records.append({'day': day, 'idx': idx, 'trigger': 'Squeeze', 'event_val': v_sqz, 'sigma': sigma, 'mode': 'volatility', **context})
        if v_can != 0:
            records.append({'day': day, 'idx': idx, 'trigger': 'Candle', 'event_val': v_can, 'sigma': sigma, 'mode': 'directional', **context})
        if v_ma != 0:
            records.append({'day': day, 'idx': idx, 'trigger': 'MA_Cross', 'event_val': v_ma, 'sigma': sigma, 'mode': 'directional', **context})
            
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
        magnitude = 0.0
        
        for p in path:
            if mode == 'volatility':
                if p >= target_price or p <= stop_price: 
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
                        
        if not hit_target and not hit_stop:
            if mode == 'volatility':
                magnitude = max(abs(path[-1] - p0), 0) / sigma
                hit_target = False
            else:
                magnitude = ((path[-1] - p0) * event_val) / sigma
                hit_target = magnitude > 0
                
        r['claim_was_true'] = int(hit_target)
        r['magnitude'] = magnitude
        event_results.append(r)
        
    return event_results

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    features_root = os.path.join(base_dir, 'DATA/ATLAS/FEATURES_5s_v2')
    l0_dir = os.path.join(features_root, 'L0')
    all_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    days = [os.path.basename(f).replace('.parquet', '') for f in all_files]
    days = [d for d in days if d.startswith('2024')]
    
    print(f"[Joint Model] Running feature extraction over {len(days)} days in 2024...")
    
    all_results = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
        for res in executor.map(process_joint_day, days):
            if res is not None:
                all_results.extend(res)
                
    df = pd.DataFrame(all_results)
    if len(df) == 0:
        print("No events found.")
        sys.exit(0)
        
    print(f"Total Event Rows Extracted: {len(df)}")
    
    # Target
    y = df['claim_was_true'].values
    
    # Features
    feature_cols = ['vwap_state', 'apz_state', 'sqz_state', 'can_state', 'ma_state']
    X = df[feature_cols].copy()
    
    # We shouldn't standardize categorical/binary states usually, but let's just train standard Logistic Regression
    clf = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True, max_iter=1000)
    clf.fit(X, y)
    
    y_pred_proba = clf.predict_proba(X)[:, 1]
    df['posterior'] = y_pred_proba
    
    # Analysis
    report_lines = []
    report_lines.append("# Joint Bayesian Model (Logistic Regression)")
    report_lines.append(f"**Total Events Trained:** {len(df)}")
    report_lines.append(f"**Base Rate (Intercept implied):** {y.mean():.4f}")
    report_lines.append("")
    report_lines.append("## 1. Feature Coefficients (Conditioned Weights)")
    report_lines.append("| Feature | Coefficient | Odds Ratio |")
    report_lines.append("|---|---|---|")
    for feat, coef in zip(feature_cols, clf.coef_[0]):
        report_lines.append(f"| {feat} | {coef:.4f} | {np.exp(coef):.4f} |")
    report_lines.append(f"| Intercept | {clf.intercept_[0]:.4f} | {np.exp(clf.intercept_[0]):.4f} |")
    report_lines.append("")
    
    report_lines.append("## 2. Posterior Tier Separation (Calibration)")
    report_lines.append("We bucket the events by their predicted posterior probability to see if confluence generates lift.")
    report_lines.append("| Tier (Percentile) | N | Mean Posterior | Actual Win Rate | Delta vs Base |")
    report_lines.append("|---|---|---|---|---|")
    
    df['tier'] = pd.qcut(df['posterior'], q=10, duplicates='drop')
    for tier, group in df.groupby('tier'):
        mean_post = group['posterior'].mean()
        actual_wr = group['claim_was_true'].mean()
        n_group = len(group)
        delta = (actual_wr - y.mean()) * 100
        report_lines.append(f"| {tier} | {n_group} | {mean_post:.4f} | {actual_wr:.4f} | {delta:+.2f} pp |")
        
    report_lines.append("")
    report_lines.append("## 3. Verdict")
    report_lines.append("If the top tier exhibits a Real > +10pp lift over the base rate, confluence provides a tradable edge.")
    
    report_path = os.path.join(base_dir, 'research', 'nt8_catalog', 'reports', 'AG_Joint_Model.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"[Joint Model] Fit complete. Report saved to {report_path}")
