import os
import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import warnings
import sys

warnings.filterwarnings("ignore")

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.abspath(os.path.join(base_dir, '..', '..')))
from core_v2.features import load_features, FEATURE_NAMES

def run_phase5():
    features_root = os.path.abspath(os.path.join(base_dir, '..', '..', 'DATA', 'ATLAS', 'FEATURES_5s_v2'))
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Priority dossiers per Doc 027
    target_dossiers = ['ATR-09', 'FIB-17', 'VA-13', 'ORDERFLOW-14']
    events_files = []
    for td in target_dossiers:
        f = glob.glob(os.path.join(base_dir, 'tests', f'*{td}*', 'events.parquet'))
        if f: events_files.extend(f)
        else:
            f2 = glob.glob(os.path.join(base_dir, 'tests', td, 'events.parquet'))
            if f2: events_files.extend(f2)
            
    print(f"Found {len(events_files)} priority dossiers to process.")
    
    final_results = []
    
    for ef in events_files:
        dossier = os.path.basename(os.path.dirname(ef))
        try:
            ev_df = pd.read_parquet(ef)
        except Exception:
            continue
            
        if len(ev_df) == 0: continue
        
        print(f"\n========================================")
        print(f"Processing {dossier}...")
        
        days = ev_df['day'].unique()
        X_list, y_list, mag_list, year_list = [], [], [], []
        
        for day in days:
            day_events = ev_df[ev_df['day'] == day]
            try:
                feats = load_features(days=[day], root=features_root, require_all=False)
            except Exception as e:
                continue
                
            for _, row in day_events.iterrows():
                idx = int(row['event_idx'])
                if idx >= len(feats): continue
                row_feats = feats.iloc[idx]
                
                # Extract V2 features safely
                vec = []
                for fn in FEATURE_NAMES:
                    if fn in row_feats:
                        val = row_feats[fn]
                        vec.append(val if pd.notnull(val) else 0.0)
                    else:
                        vec.append(0.0)
                        
                X_list.append(vec)
                y_list.append(row['hit'])
                mag_list.append(row['magnitude'])
                year_list.append(day[:4])
                
        if len(X_list) < 100:
            print(f"Not enough valid V2 events for {dossier}.")
            continue
            
        X = np.array(X_list)
        y = np.array(y_list)
        mag = np.array(mag_list)
        years = np.array(year_list)
        
        mask_2024 = (years == '2024')
        mask_2025 = (years == '2025')
        
        X_train, y_train = X[mask_2024], y[mask_2024]
        if len(X_train) < 50:
            print(f"Not enough 2024 training data for {dossier}.")
            continue
            
        # Step 1: L1 Feature Selection on 2024
        # Standardize
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0) + 1e-9
        X_train_s = (X_train - X_mean) / X_std
        
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.05, max_iter=200)
        clf.fit(X_train_s, y_train)
        
        coefs = clf.coef_[0]
        selected_idx = np.where(np.abs(coefs) > 1e-4)[0]
        
        if len(selected_idx) == 0:
            print("L1 selection dropped all features. Using top 5 by correlation.")
            corrs = [np.abs(np.corrcoef(X_train_s[:, i], y_train)[0, 1]) for i in range(X_train_s.shape[1])]
            corrs = np.nan_to_num(corrs)
            selected_idx = np.argsort(corrs)[-5:]
            
        print(f"Selected {len(selected_idx)} features:")
        for idx in selected_idx:
            print(f" - {FEATURE_NAMES[idx]}")
            
        # Step 2: Fit Statsmodels Logit on selected features
        X_train_sel = X_train_s[:, selected_idx]
        X_train_sm = np.column_stack((np.ones(len(X_train_sel)), X_train_sel))
        
        used_sm = True
        try:
            model = sm.Logit(y_train, X_train_sm).fit(disp=0)
        except Exception as e:
            print(f"Logit failed: {e}. Falling back to SKLearn.")
            model = LogisticRegression(penalty='none').fit(X_train_sel, y_train)
            used_sm = False
            
        # Predict 2024
        if used_sm:
            p_2024 = model.predict(X_train_sm)
        else:
            p_2024 = model.predict_proba(X_train_sel)[:, 1]
            
        # Choose thresholds: top 15% and bottom 15% on 2024
        p_hi = np.percentile(p_2024, 85)
        p_lo = np.percentile(p_2024, 15)
        
        # Step 3: Evaluate on 2025
        X_test, y_test, mag_test = X[mask_2025], y[mask_2025], mag[mask_2025]
        if len(X_test) < 10:
            print(f"Not enough 2025 test data for {dossier}.")
            continue
            
        X_test_s = (X_test - X_mean) / X_std
        X_test_sel = X_test_s[:, selected_idx]
        X_test_sm = np.column_stack((np.ones(len(X_test_sel)), X_test_sel))
        
        if used_sm:
            p_2025 = model.predict(X_test_sm)
        else:
            p_2025 = model.predict_proba(X_test_sel)[:, 1]
            
        # Calculate EV and bootstrap CI for 2025
        def calc_metrics(y_prob, y_true, mags, p_hi, p_lo):
            act_mask = y_prob >= p_hi
            inv_mask = y_prob <= p_lo
            
            act_n = np.sum(act_mask)
            act_ev = np.sum(mags[act_mask]) / max(1, act_n)
            act_wr = np.mean(y_true[act_mask]) if act_n > 0 else 0
            
            inv_n = np.sum(inv_mask)
            inv_ev = np.sum(-mags[inv_mask]) / max(1, inv_n)
            inv_wr = np.mean(1 - y_true[inv_mask]) if inv_n > 0 else 0
            
            return act_n, act_ev, act_wr, inv_n, inv_ev, inv_wr, mags[act_mask], -mags[inv_mask]

        act_n, act_ev, act_wr, inv_n, inv_ev, inv_wr, act_mags, inv_mags = calc_metrics(p_2025, y_test, mag_test, p_hi, p_lo)
        
        # Bootstrap EV CI
        B = 1000
        act_ev_boot = []
        inv_ev_boot = []
        for _ in range(B):
            if act_n > 0:
                boot_idx = np.random.choice(len(act_mags), size=act_n, replace=True)
                act_ev_boot.append(np.mean(act_mags[boot_idx]))
            if inv_n > 0:
                boot_idx = np.random.choice(len(inv_mags), size=inv_n, replace=True)
                inv_ev_boot.append(np.mean(inv_mags[boot_idx]))
                
        act_ci_lo = np.percentile(act_ev_boot, 2.5) if act_ev_boot else 0
        inv_ci_lo = np.percentile(inv_ev_boot, 2.5) if inv_ev_boot else 0
        
        act_mode = pd.Series(np.round(act_mags, 0)).mode().values[0] if act_n > 0 else 0
        inv_mode = pd.Series(np.round(inv_mags, 0)).mode().values[0] if inv_n > 0 else 0
        
        act_valid = (act_ci_lo > 0) and (act_mode >= 2.0)
        inv_valid = (inv_ci_lo > 0) and (inv_mode >= 2.0)
        
        print(f"\n2025 Evaluation:")
        print(f"ACT Branch  (P >= {p_hi:.3f}): N={act_n}, WR={act_wr:.2f}, EV={act_ev:.2f} pts (CI_lo: {act_ci_lo:.2f}), Mode={act_mode} pts | Valid={act_valid}")
        print(f"INV Branch  (P <= {p_lo:.3f}): N={inv_n}, WR={inv_wr:.2f}, EV={inv_ev:.2f} pts (CI_lo: {inv_ci_lo:.2f}), Mode={inv_mode} pts | Valid={inv_valid}")
        
        final_results.append({
            'dossier': dossier,
            'features': len(selected_idx),
            'act_n': act_n,
            'act_ev': act_ev,
            'act_valid': act_valid,
            'inv_n': inv_n,
            'inv_ev': inv_ev,
            'inv_valid': inv_valid
        })

    # Write Markdown summary
    md = "# Phase-5 F-Space Logistic Evaluation\n\n"
    md += "Three-way policy evaluation (ACT / SKIP / INVERT) using V2 Features (Doc 027).\n\n"
    md += "| Dossier | N_Feats | ACT N | ACT EV (pts) | ACT Valid? | INV N | INV EV (pts) | INV Valid? |\n"
    md += "|---------|---------|-------|--------------|------------|-------|--------------|------------|\n"
    for r in final_results:
        md += f"| {r['dossier']} | {r['features']} | {r['act_n']} | {r['act_ev']:.2f} | {'✅' if r['act_valid'] else '❌'} | {r['inv_n']} | {r['inv_ev']:.2f} | {'✅' if r['inv_valid'] else '❌'} |\n"
        
    with open(os.path.join(reports_dir, 'AG_cat_00_PHASE5.md'), 'w') as f:
        f.write(md)
        
    print("\nPhase 5 Evaluation Complete. Report saved to reports/AG_cat_00_PHASE5.md")

if __name__ == '__main__':
    run_phase5()
