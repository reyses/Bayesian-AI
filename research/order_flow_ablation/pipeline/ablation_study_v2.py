import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def run_ablation_v2():
    print("Loading baseline features...")
    baseline_file = "DATA/ATLAS/baseline_features_416D.parquet"
    if not os.path.exists(baseline_file):
        print(f"Baseline features not found: {baseline_file}")
        return
        
    baseline_df = pd.read_parquet(baseline_file)
    
    print("Loading delta features...")
    delta_file = "DATA/ATLAS/order_flow_delta_5s.parquet"
    if not os.path.exists(delta_file):
        print("Delta features not found.")
        return
        
    delta_df = pd.read_parquet(delta_file)
    
    # Ensure timezone awareness matches
    if baseline_df.index.tz is None and delta_df.index.tz is not None:
        baseline_df.index = baseline_df.index.tz_localize('UTC')
    elif baseline_df.index.tz is not None and delta_df.index.tz is None:
        delta_df.index = delta_df.index.tz_localize('UTC')
        
    print("Merging datasets...")
    merged_df = baseline_df.join(delta_df[['cum_delta', 'divergence']], how='inner')
    
    print("Computing baseline volatility...")
    # 5s returns, rolling standard deviation over 10 minutes (120 bars)
    ret = merged_df['close'].diff()
    merged_df['vol_5s'] = ret.rolling(window=120, min_periods=30).std()
    
    # Drop rows without vol
    merged_df.dropna(subset=['vol_5s'], inplace=True)
    
    horizons = [3, 6, 12, 24, 60]  # 15s, 30s, 1m, 2m, 5m
    
    non_feature_cols = ['target', 'cum_delta', 'divergence', 'delta', 'volume', 'open', 'high', 'low', 'close', 'vol_5s']
    baseline_cols = [c for c in merged_df.columns if c not in non_feature_cols]
    delta_features = ['cum_delta', 'divergence']
    
    results = []
    
    for k in horizons:
        print(f"\n--- Preparing horizon k={k} bars ---")
        df_k = merged_df[['close', 'vol_5s', 'cum_delta', 'divergence'] + baseline_cols].copy()
        
        # Vol-normalized thresholds
        # R_k: reversal threshold (1.5 * vol * sqrt(k))
        # deadband: flat filtering (0.5 * vol * sqrt(k))
        R_k = 1.5 * df_k['vol_5s'] * np.sqrt(k)
        deadband = 0.5 * df_k['vol_5s'] * np.sqrt(k)
        
        # Forward window calculations
        # shift(-k) pulls t+k to t. rolling(k).min() computes min over [t+1, t+k]
        fwd_min = df_k['close'].shift(-k).rolling(window=k, min_periods=k).min()
        fwd_max = df_k['close'].shift(-k).rolling(window=k, min_periods=k).max()
        fwd_price = df_k['close'].shift(-k)
        
        delta_P = fwd_price - df_k['close']
        prior_move = df_k['close'] - df_k['close'].shift(k)
        
        # Drop rows where future data is NaN (end of dataset)
        valid_idx = fwd_price.notna() & fwd_min.notna() & prior_move.notna()
        
        # --- Target 1: Direction ---
        dir_mask = valid_idx & (delta_P.abs() > deadband)
        df_dir = df_k[dir_mask].copy()
        y_dir = (delta_P[dir_mask] > 0).astype(int)
        
        # --- Target 2: Exhaustion ---
        exh_mask = valid_idx & (prior_move.abs() > R_k)
        df_exh = df_k[exh_mask].copy()
        
        # Exhaustion logic
        prior_up = prior_move[exh_mask] > 0
        prior_dn = prior_move[exh_mask] < 0
        
        # If prior up, did it drop by R_k?
        reversal_down = fwd_min[exh_mask] <= (df_exh['close'] - R_k[exh_mask])
        # If prior down, did it rally by R_k?
        reversal_up = fwd_max[exh_mask] >= (df_exh['close'] + R_k[exh_mask])
        
        y_exh = np.where(prior_up, reversal_down, reversal_up).astype(int)
        
        targets = [
            ('Direction', df_dir, y_dir),
            ('Exhaustion', df_exh, pd.Series(y_exh, index=df_exh.index))
        ]
        
        for t_name, df_t, y_t in targets:
            print(f"Evaluating {t_name} at k={k} (N={len(df_t)})...")
            if len(df_t) < 5000:
                print("Not enough samples, skipping.")
                continue
                
            split_idx = int(len(df_t) * 0.66)
            
            X_train_base = df_t[baseline_cols].iloc[:split_idx]
            y_train = y_t.iloc[:split_idx]
            X_test_base = df_t[baseline_cols].iloc[split_idx:]
            y_test = y_t.iloc[split_idx:]
            
            # Check class balance
            if y_train.mean() < 0.05 or y_train.mean() > 0.95:
                print("Classes too imbalanced, skipping.")
                continue
                
            model_base = XGBClassifier(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42, tree_method='hist')
            model_base.fit(X_train_base, y_train)
            
            preds_base = model_base.predict_proba(X_test_base)[:, 1]
            auc_base = roc_auc_score(y_test, preds_base)
            
            print(f"  Baseline AUC: {auc_base:.4f}")
            
            if auc_base < 0.515:
                print("  Failed Pre-Gate (No Baseline Skill). Dropping cell.")
                results.append({'target': t_name, 'k': k, 'N': len(y_test), 'auc_base': auc_base, 'auc_delta': None, 'lift': None})
                continue
                
            print("  Pre-Gate Passed. Training Baseline+Delta...")
            X_train_delta = df_t[baseline_cols + delta_features].iloc[:split_idx]
            X_test_delta = df_t[baseline_cols + delta_features].iloc[split_idx:]
            
            model_delta = XGBClassifier(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42, tree_method='hist')
            model_delta.fit(X_train_delta, y_train)
            
            preds_delta = model_delta.predict_proba(X_test_delta)[:, 1]
            auc_delta = roc_auc_score(y_test, preds_delta)
            lift = auc_delta - auc_base
            
            print(f"  Baseline+Delta AUC: {auc_delta:.4f} (Lift: {lift:+.4f})")
            
            # Sub-period stability
            test_df_results = pd.DataFrame({'y_true': y_test, 'pred': preds_delta}, index=y_test.index)
            test_groups = test_df_results.groupby(pd.Grouper(freq='4W'))
            
            stability = []
            for period, group in test_groups:
                if len(group) > 500 and group['y_true'].nunique() > 1:
                    period_auc = roc_auc_score(group['y_true'], group['pred'])
                    stability.append(f"{period.date()}: {period_auc:.4f}")
            
            results.append({
                'target': t_name,
                'k': k,
                'N': len(y_test),
                'auc_base': auc_base,
                'auc_delta': auc_delta,
                'lift': lift,
                'stability': " | ".join(stability)
            })
            
            # Print importances
            importances = pd.Series(model_delta.feature_importances_, index=X_train_delta.columns)
            print("  Top 5 Features:")
            print(importances.sort_values(ascending=False).head(5))
            
    # Generate final report
    out_dir = "research/order_flow_ablation/reports"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "delta_verdict_v2.md"), "w") as f:
        f.write("# Order Flow Ablation V2 Verdict\n\n")
        f.write("| Target | Horizon (bars) | N (Test) | Base AUC | Delta AUC | Lift | Sub-Period Stability |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for r in results:
            if r['auc_delta'] is not None:
                f.write(f"| {r['target']} | {r['k']} | {r['N']} | {r['auc_base']:.4f} | {r['auc_delta']:.4f} | **{r['lift']:+.4f}** | {r['stability']} |\n")
            else:
                f.write(f"| {r['target']} | {r['k']} | {r['N']} | {r['auc_base']:.4f} | N/A | N/A | Failed Pre-Gate |\n")

if __name__ == '__main__':
    run_ablation_v2()
