import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from scipy.fft import fft, ifft

def make_fourier_surrogate(series):
    """Generate a Fourier phase-randomized surrogate preserving the power spectrum."""
    x = series.values
    X = fft(x)
    n = len(X)
    
    # Random phases
    phases = np.random.uniform(0, 2*np.pi, n)
    # Ensure conjugate symmetry for real signal
    phases[0] = 0
    if n % 2 == 0:
        phases[n//2] = 0
        phases[n//2 + 1:] = -phases[1:n//2][::-1]
    else:
        phases[n//2 + 1:] = -phases[1:n//2 + 1][::-1]
        
    # Apply phases
    X_surr = np.abs(X) * np.exp(1j * phases)
    x_surr = np.real(ifft(X_surr))
    
    # Rescale to match variance exactly
    x_surr = (x_surr - np.mean(x_surr)) / np.std(x_surr) * np.std(x) + np.mean(x)
    return x_surr

def run_stage_1():
    print("Stage 1: Inflection Ablation (Make-or-Break)")
    baseline_file = "DATA/ATLAS/baseline_features_416D.parquet"
    delta_file = "DATA/ATLAS/order_flow_delta_5s.parquet"
    
    if not os.path.exists(baseline_file) or not os.path.exists(delta_file):
        print("Missing required datasets.")
        return
        
    baseline_df = pd.read_parquet(baseline_file)
    delta_df = pd.read_parquet(delta_file)
    
    if baseline_df.index.tz is None and delta_df.index.tz is not None:
        baseline_df.index = baseline_df.index.tz_localize('UTC')
    elif baseline_df.index.tz is not None and delta_df.index.tz is None:
        delta_df.index = delta_df.index.tz_localize('UTC')
        
    merged_df = baseline_df.join(delta_df[['cum_delta', 'divergence']], how='inner')
    
    ret = merged_df['close'].diff()
    merged_df['vol_5s'] = ret.rolling(window=120, min_periods=30).std()
    merged_df.dropna(subset=['vol_5s'], inplace=True)
    
    horizons = [60]  # Focus on 5m horizon for the Make-or-Break test
    
    non_feature_cols = ['target', 'cum_delta', 'divergence', 'delta', 'volume', 'open', 'high', 'low', 'close', 'vol_5s']
    baseline_cols = [c for c in merged_df.columns if c not in non_feature_cols]
    delta_features = ['cum_delta', 'divergence']
    
    results = []
    
    for k in horizons:
        print(f"\n--- Preparing horizon k={k} bars ---")
        df_k = merged_df[['close', 'vol_5s', 'cum_delta', 'divergence'] + baseline_cols].copy()
        
        R_k = 1.5 * df_k['vol_5s'] * np.sqrt(k)
        
        fwd_min = df_k['close'].shift(-k).rolling(window=k, min_periods=k).min()
        fwd_max = df_k['close'].shift(-k).rolling(window=k, min_periods=k).max()
        fwd_price = df_k['close'].shift(-k)
        
        prior_move = df_k['close'] - df_k['close'].shift(k)
        
        valid_idx = fwd_price.notna() & fwd_min.notna() & prior_move.notna()
        
        # Exhaustion Target (Inflection Timing)
        exh_mask = valid_idx & (prior_move.abs() > R_k)
        df_exh = df_k[exh_mask].copy()
        
        prior_up = prior_move[exh_mask] > 0
        prior_dn = prior_move[exh_mask] < 0
        
        reversal_down = fwd_min[exh_mask] <= (df_exh['close'] - R_k[exh_mask])
        reversal_up = fwd_max[exh_mask] >= (df_exh['close'] + R_k[exh_mask])
        
        y_exh = np.where(prior_up, reversal_down, reversal_up).astype(int)
        
        print(f"Evaluating Exhaustion at k={k} (N={len(df_exh)})...")
        split_idx = int(len(df_exh) * 0.66)
        
        X_train_base = df_exh[baseline_cols].iloc[:split_idx]
        y_train = pd.Series(y_exh[:split_idx])
        X_test_base = df_exh[baseline_cols].iloc[split_idx:]
        y_test = pd.Series(y_exh[split_idx:])
        
        # 1. Baseline Model
        model_base = XGBClassifier(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42, tree_method='hist')
        model_base.fit(X_train_base, y_train)
        preds_base = model_base.predict_proba(X_test_base)[:, 1]
        auc_base = roc_auc_score(y_test, preds_base)
        print(f"  Baseline AUC: {auc_base:.4f}")
        
        if auc_base < 0.52 or auc_base > 0.85:
            print("  Failed Pre-Gate (AUC out of bounds). Breaking.")
            break
            
        # 2. Baseline + True Delta
        X_train_delta = df_exh[baseline_cols + delta_features].iloc[:split_idx]
        X_test_delta = df_exh[baseline_cols + delta_features].iloc[split_idx:]
        model_delta = XGBClassifier(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42, tree_method='hist')
        model_delta.fit(X_train_delta, y_train)
        preds_delta = model_delta.predict_proba(X_test_delta)[:, 1]
        auc_delta = roc_auc_score(y_test, preds_delta)
        lift = auc_delta - auc_base
        print(f"  Baseline+Delta AUC: {auc_delta:.4f} (Lift: {lift:+.4f})")
        
        # 3. Fourier Phase-Randomized Null Test
        print("  Running Fourier Phase-Randomized Null Test (N=10 seeds)...")
        null_aucs = []
        # Create a single surrogate dataframe
        surrogate_train = X_train_delta.copy()
        surrogate_test = X_test_delta.copy()
        
        for seed in range(10):
            np.random.seed(seed)
            # Randomize the delta features
            for feat in delta_features:
                surrogate_train[feat] = make_fourier_surrogate(X_train_delta[feat])
                surrogate_test[feat] = make_fourier_surrogate(X_test_delta[feat])
                
            null_model = XGBClassifier(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42, tree_method='hist')
            null_model.fit(surrogate_train, y_train)
            null_preds = null_model.predict_proba(surrogate_test)[:, 1]
            null_aucs.append(roc_auc_score(y_test, null_preds))
            
        p95_null = np.percentile(null_aucs, 95)
        print(f"  Fourier Null 95th Percentile AUC: {p95_null:.4f}")
        passed_null = auc_delta > p95_null
        print(f"  Passed Null Test: {passed_null}")
        
        # Output verdict
        out_dir = "research/order_flow_ablation/reports"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "stage_1_inflection_verdict.md"), "w") as f:
            f.write("# Stage 1: Inflection Ablation Verdict\n\n")
            f.write(f"- **Baseline AUC:** {auc_base:.4f} (In Bounds: {0.52 <= auc_base <= 0.85})\n")
            f.write(f"- **True Delta AUC:** {auc_delta:.4f} (Lift: {lift:+.4f})\n")
            f.write(f"- **Fourier Null 95th:** {p95_null:.4f}\n")
            f.write(f"- **Verdict (MAKE/BREAK):** {'MAKE' if passed_null and lift > 0 else 'BREAK'}\n")
            
        break

if __name__ == '__main__':
    run_stage_1()
