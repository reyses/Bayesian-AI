import os
import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

def run_phase5_final():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    target_dossiers = ['ATR-09', 'FIB-17', 'VA-13', 'ORDERFLOW-14']
    final_results = []
    
    for td in target_dossiers:
        dossier_path = os.path.join(base_dir, 'tests', td)
        if not os.path.exists(dossier_path):
            matched = glob.glob(os.path.join(base_dir, 'tests', f'*{td}*'))
            if matched:
                dossier_path = matched[0]
            else:
                continue
                
        try:
            X_Phe = np.load(os.path.join(dossier_path, 'X_Phe.npy'))
            X_PhXit = np.load(os.path.join(dossier_path, 'X_PhXit.npy'))
            X_PhPost = np.load(os.path.join(dossier_path, 'X_PhPost.npy'))
            Y = np.load(os.path.join(dossier_path, 'Y.npy'))
            Mags = np.load(os.path.join(dossier_path, 'Mags.npy'))
            Years = np.load(os.path.join(dossier_path, 'Years.npy'))
        except Exception as e:
            print(f"Skipping {td}: missing numpy files. {e}")
            continue
            
        print(f"\n========================================")
        print(f"Processing {td}...")
        print(f"Lengths: X={len(X_Phe)}, Y={len(Y)}, Years={len(Years)}")
        
        X = np.concatenate([X_Phe, X_PhXit, X_PhPost], axis=1).astype(float)
        y = np.array(Y).astype(int)
        mag = np.array(Mags).astype(float)
        years = np.array(Years).astype(str)
        
        # Dynamic train/test year split
        unique_years, counts = np.unique(years, return_counts=True)
        train_year, test_year = None, None
        
        for i, yr in enumerate(unique_years):
            if counts[i] >= 30:
                train_year = yr
                if i + 1 < len(unique_years):
                    test_year = unique_years[i + 1]
                break
                
        if train_year is None or test_year is None:
            print(f"Not enough sequential years with data for {td}.")
            continue
            
        print(f"Training on {train_year}, Testing on {test_year}")
        mask_train = (years == train_year)
        mask_test = (years == test_year)
        
        X_train, y_train = X[mask_train], y[mask_train]
            
        # Step 1: L1 Feature Selection on 2024
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
            
        print(f"Selected {len(selected_idx)} features out of {X.shape[1]}")
        
        # Step 2: Fit Statsmodels Logit
        X_train_sel = X_train_s[:, selected_idx]
        X_train_sm = np.column_stack((np.ones(len(X_train_sel)), X_train_sel))
        
        used_sm = True
        try:
            model = sm.Logit(y_train, X_train_sm).fit(disp=0)
        except Exception as e:
            print(f"Logit failed: {e}. Falling back to SKLearn.")
            model = LogisticRegression(penalty=None).fit(X_train_sel, y_train)
            used_sm = False
            
        if used_sm:
            p_2024 = model.predict(X_train_sm)
        else:
            p_2024 = model.predict_proba(X_train_sel)[:, 1]
            
        p_hi = np.percentile(p_2024, 85)
        p_lo = np.percentile(p_2024, 15)
        
        # Step 3: Evaluate on test_year
        X_test, y_test, mag_test = X[mask_test], y[mask_test], mag[mask_test]
        if len(X_test) < 10:
            print(f"Not enough {test_year} test data for {td}.")
            continue
            
        X_test_s = (X_test - X_mean) / X_std
        X_test_sel = X_test_s[:, selected_idx]
        X_test_sm = np.column_stack((np.ones(len(X_test_sel)), X_test_sel))
        
        if used_sm:
            p_2025 = model.predict(X_test_sm)
        else:
            p_2025 = model.predict_proba(X_test_sel)[:, 1]
            
        # calc_metrics
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
        
        B = 1000
        act_ev_boot, inv_ev_boot = [], []
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
            'dossier': td,
            'features': len(selected_idx),
            'act_n': act_n,
            'act_ev': act_ev,
            'act_valid': act_valid,
            'inv_n': inv_n,
            'inv_ev': inv_ev,
            'inv_valid': inv_valid
        })

    md = "# Phase-5 F-Space Logistic Evaluation\n\n"
    md += "Three-way policy evaluation (ACT / SKIP / INVERT) using V2 Features (Doc 027).\n\n"
    md += "| Dossier | N_Feats | ACT N | ACT EV (pts) | ACT Valid? | INV N | INV EV (pts) | INV Valid? |\n"
    md += "|---------|---------|-------|--------------|------------|-------|--------------|------------|\n"
    for r in final_results:
        md += f"| {r['dossier']} | {r['features']} | {r['act_n']} | {r['act_ev']:.2f} | {'✅' if r['act_valid'] else '❌'} | {r['inv_n']} | {r['inv_ev']:.2f} | {'✅' if r['inv_valid'] else '❌'} |\n"
        
    with open(os.path.join(reports_dir, 'AG_cat_00_PHASE5.md'), 'w', encoding='utf-8') as f:
        f.write(md)
        
    print("\nPhase 5 Evaluation Complete. Report saved to reports/AG_cat_00_PHASE5.md")

if __name__ == '__main__':
    run_phase5_final()
