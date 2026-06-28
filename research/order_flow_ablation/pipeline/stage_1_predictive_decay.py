import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.fft import fft, ifft

def make_fourier_surrogate(series):
    """Generate a Fourier phase-randomized surrogate preserving the power spectrum."""
    x = series.values
    X = fft(x)
    n = len(X)
    
    phases = np.random.uniform(0, 2*np.pi, n)
    phases[0] = 0
    if n % 2 == 0:
        phases[n//2] = 0
        phases[n//2 + 1:] = -phases[1:n//2][::-1]
    else:
        phases[n//2 + 1:] = -phases[1:n//2 + 1][::-1]
        
    X_surr = np.abs(X) * np.exp(1j * phases)
    x_surr = np.real(ifft(X_surr))
    
    # Avoid div by zero in flat arrays
    std_x = np.std(x)
    if std_x == 0:
        return x
        
    x_surr = (x_surr - np.mean(x_surr)) / np.std(x_surr) * std_x + np.mean(x)
    return x_surr

def run_stage_1_decay():
    print("Stage 1: Predictive Ablation & Causal Decay")
    
    # We will run this on the 1m resampled data to get stable forward bars
    print("Loading 5s Order Flow Delta dataset and resampling to 1m...")
    df_raw = pd.read_parquet("DATA/ATLAS/order_flow_delta_5s.parquet")
    if df_raw.index.tz is None:
        df_raw.index = df_raw.index.tz_localize('UTC')
        
    df = df_raw.resample('1min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'delta': 'sum'
    }).dropna()

    # Calculate structural features
    df['price_delta'] = df['close'] - df['open']
    
    df['wick_from_close'] = np.where(
        df['price_delta'] > 0,
        df['high'] - df['close'],
        df['close'] - df['low']
    )
    
    epsilon = 1e-8
    df['total_range'] = df['high'] - df['low']
    df['body_vol'] = (df['price_delta'] / (df['total_range'] + epsilon)) * df['volume']
    df['wick_close_vol'] = (df['wick_from_close'] / (df['total_range'] + epsilon)) * df['volume']
    
    # Volatility baseline feature
    df['vol_60m'] = df['close'].diff().rolling(window=60, min_periods=10).std()
    
    df.dropna(inplace=True)
    df = df[df['volume'] > 0].copy()
    
    baseline_features = ['vol_60m', 'price_delta', 'total_range']
    new_features = ['delta', 'body_vol', 'wick_close_vol']
    all_features = baseline_features + new_features
    
    max_k = 10
    results = []
    
    split_idx = int(len(df) * 0.66)
    
    for k in range(1, max_k + 1):
        print(f"\n--- Training for Forward Horizon: t+{k} bars ---")
        
        # Target: direction of the k-th forward bar
        # e.g., if k=1, return is close_t+1 - close_t
        # e.g., if k=2, return is close_t+2 - close_t+1
        df_k = df.copy()
        
        # Calculate exactly the return of the k-th bar
        df_k['fwd_return_k'] = df_k['close'].shift(-k) - df_k['close'].shift(-(k-1))
        df_k.dropna(subset=['fwd_return_k'], inplace=True)
        
        # Binary target: 1 if UP, 0 if DOWN
        y = (df_k['fwd_return_k'] > 0).astype(int)
        
        X_train = df_k[all_features].iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = df_k[all_features].iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        # 1. Baseline Model
        model_base = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=42, tree_method='hist')
        model_base.fit(X_train[baseline_features], y_train)
        preds_base = model_base.predict_proba(X_test[baseline_features])[:, 1]
        auc_base = roc_auc_score(y_test, preds_base)
        acc_base = accuracy_score(y_test, preds_base > 0.5)
        
        # 2. Full Model (Baseline + Absorption Features)
        model_full = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=42, tree_method='hist')
        model_full.fit(X_train, y_train)
        preds_full = model_full.predict_proba(X_test)[:, 1]
        auc_full = roc_auc_score(y_test, preds_full)
        acc_full = accuracy_score(y_test, preds_full > 0.5)
        
        # 3. Fourier Phase-Randomized Null
        # Randomize ONLY the new structural features (keep baseline intact to prove the lift is real)
        np.random.seed(42 + k)
        X_train_null = X_train.copy()
        X_test_null = X_test.copy()
        
        for feat in new_features:
            X_train_null[feat] = make_fourier_surrogate(X_train[feat])
            X_test_null[feat] = make_fourier_surrogate(X_test[feat])
            
        model_null = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=42, tree_method='hist')
        model_null.fit(X_train_null, y_train)
        preds_null = model_null.predict_proba(X_test_null)[:, 1]
        auc_null = roc_auc_score(y_test, preds_null)
        acc_null = accuracy_score(y_test, preds_null > 0.5)
        
        print(f"  Baseline AUC: {auc_base:.4f} | Acc: {acc_base:.4f}")
        print(f"  Full Mod AUC: {auc_full:.4f} | Acc: {acc_full:.4f}")
        print(f"  Null Mod AUC: {auc_null:.4f} | Acc: {acc_null:.4f}")
        
        results.append({
            'k': k,
            'Baseline AUC': auc_base,
            'Full Model AUC': auc_full,
            'Null Model AUC': auc_null,
            'Baseline Acc': acc_base,
            'Full Model Acc': acc_full,
            'Null Model Acc': acc_null
        })
        
    res_df = pd.DataFrame(results).set_index('k')
    
    print("\n--- Predictive Decay Results ---")
    print(res_df.to_string())
    
    # Plotting
    os.makedirs('research/order_flow_ablation/reports', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # AUC Plot
    ax1.plot(res_df.index, res_df['Full Model AUC'], marker='o', label='Full Model (Absorption Edge)', color='#2ecc71', linewidth=2)
    ax1.plot(res_df.index, res_df['Baseline AUC'], marker='s', label='Baseline (Vol/Price)', color='#95a5a6', linestyle='--')
    ax1.plot(res_df.index, res_df['Null Model AUC'], marker='x', label='Fourier Null', color='#e74c3c', linestyle=':')
    ax1.axhline(0.5, color='black', linewidth=1)
    ax1.set_title('Probability Decay (AUC) over Forward 1m Bars')
    ax1.set_xlabel('Forward Bar (k)')
    ax1.set_ylabel('ROC AUC')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy Plot
    ax2.plot(res_df.index, res_df['Full Model Acc'], marker='o', label='Full Model', color='#2ecc71', linewidth=2)
    ax2.plot(res_df.index, res_df['Baseline Acc'], marker='s', label='Baseline', color='#95a5a6', linestyle='--')
    ax2.plot(res_df.index, res_df['Null Model Acc'], marker='x', label='Fourier Null', color='#e74c3c', linestyle=':')
    ax2.axhline(0.5, color='black', linewidth=1)
    ax2.set_title('Accuracy Decay over Forward 1m Bars')
    ax2.set_xlabel('Forward Bar (k)')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('research/order_flow_ablation/reports/stage_1_decay_curve.png')
    print("\nSaved decay curve to: research/order_flow_ablation/reports/stage_1_decay_curve.png")

if __name__ == "__main__":
    run_stage_1_decay()
