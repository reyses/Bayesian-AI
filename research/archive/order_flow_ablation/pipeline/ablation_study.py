import os
import glob
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import joblib

def run_ablation():
    print("Loading baseline features...")
    # Load a sample of FEATURES_5s_v2 (e.g. one parquet file) or the whole thing if it fits in memory
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
    
    # Ensure timezone awareness matches (assume both are UTC localized)
    if baseline_df.index.tz is None and delta_df.index.tz is not None:
        baseline_df.index = baseline_df.index.tz_localize('UTC')
    elif baseline_df.index.tz is not None and delta_df.index.tz is None:
        delta_df.index = delta_df.index.tz_localize('UTC')
        
    print("Merging datasets...")
    # Inner join on index
    merged_df = baseline_df.join(delta_df[['cum_delta', 'divergence', 'delta', 'volume']], how='inner')
    
    print("Computing target...")
    # Forward 30m return (30m / 5s = 360 bars)
    merged_df['target'] = merged_df['close'].shift(-360) - merged_df['close']
    merged_df.dropna(subset=['target'], inplace=True)
    
    # Split: Strict temporal walk-forward
    # First 4 months train, last 2 months test.
    # We find the splitting index at 66% of the data.
    split_idx = int(len(merged_df) * 0.66)
    
    train_df = merged_df.iloc[:split_idx].copy()
    test_df = merged_df.iloc[split_idx:].copy()
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Baseline features (416D). Let's extract columns that belong to the v2 grid.
    # Usually they are named like '1D_...' or we can just drop the specific delta/target columns
    non_feature_cols = ['target', 'cum_delta', 'divergence', 'delta', 'volume', 'open', 'high', 'low', 'close']
    baseline_cols = [c for c in train_df.columns if c not in non_feature_cols]
    
    X_train_base = train_df[baseline_cols]
    y_train = train_df['target']
    
    X_test_base = test_df[baseline_cols]
    y_test = test_df['target']
    
    print("Training Baseline XGBoost...")
    model_base = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42, tree_method='hist')
    model_base.fit(X_train_base, y_train)
    
    preds_base = model_base.predict(X_test_base)
    r2_base = r2_score(y_test, preds_base)
    print(f"Baseline R2: {r2_base:.4f}")
    
    print("Training Baseline + Delta XGBoost...")
    delta_features = ['cum_delta', 'divergence']
    X_train_delta = train_df[baseline_cols + delta_features]
    X_test_delta = test_df[baseline_cols + delta_features]
    
    model_delta = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42, tree_method='hist')
    model_delta.fit(X_train_delta, y_train)
    
    preds_delta = model_delta.predict(X_test_delta)
    r2_delta = r2_score(y_test, preds_delta)
    print(f"Baseline+Delta R2: {r2_delta:.4f}")
    
    # Feature Importance
    importances = pd.Series(model_delta.feature_importances_, index=X_train_delta.columns)
    print("Top 10 features in Delta model:")
    top_10 = importances.sort_values(ascending=False).head(10)
    print(top_10)
    
    # Sub-period stability check
    print("\nRunning Sub-period Stability Check on Test Set...")
    # Group test_df into roughly 4-week chunks
    test_df['pred_delta'] = preds_delta
    # Since index is datetime, we can group by Month or 4-week periods
    test_groups = test_df.groupby(pd.Grouper(freq='4W'))
    
    stability_results = []
    for period, group in test_groups:
        if len(group) < 1000:
            continue
        period_r2 = r2_score(group['target'], group['pred_delta'])
        res = f"Period {period.date()}: R2 = {period_r2:.4f} (N={len(group)})"
        print(res)
        stability_results.append(res)
    
    # Save results for reporting
    out_dir = "research/order_flow_ablation/reports"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ablation_results.txt"), "w") as f:
        f.write(f"Baseline R2: {r2_base:.4f}\n")
        f.write(f"Baseline+Delta R2: {r2_delta:.4f}\n")
        f.write(f"Improvement: {(r2_delta - r2_base):.4f}\n")
        f.write("\nTop 10 features:\n")
        f.write(top_10.to_string())
        f.write("\n\nSub-period Stability:\n")
        f.write("\n".join(stability_results))

if __name__ == '__main__':
    run_ablation()
