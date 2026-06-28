import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from scipy.fft import fft, ifft
from datetime import datetime

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
    
    x_surr = (x_surr - np.mean(x_surr)) / (np.std(x_surr) + 1e-9) * np.std(x) + np.mean(x)
    return x_surr

def run_stage_1_layered():
    print("Stage 1: 3-Layer Inflection Ablation (Exhaustion Target)")
    
    # 1. Load Data
    baseline_file = "DATA/ATLAS/baseline_features_416D.parquet"
    delta_file = "DATA/ATLAS/order_flow_delta_5s.parquet"
    
    baseline_df = pd.read_parquet(baseline_file)
    delta_df = pd.read_parquet(delta_file)
    
    if baseline_df.index.tz is None and delta_df.index.tz is not None:
        baseline_df.index = baseline_df.index.tz_localize('UTC')
    elif baseline_df.index.tz is not None and delta_df.index.tz is None:
        delta_df.index = delta_df.index.tz_localize('UTC')
        
    merged_df = baseline_df.join(delta_df[['cum_delta', 'divergence', 'delta', 'volume', 'open', 'high', 'low', 'close']], how='inner', rsuffix='_trade')
    
    # 2. Apply P0 Reconciliation Gate (Filter matched dates only)
    # The P0 script identified mismatch on 2025 roll weeks.
    bad_dates = [
        '2025-09-14', '2025-09-15', '2025-09-16', '2025-09-17', '2025-09-18', '2025-09-19', '2025-09-21', 
        '2025-09-22', '2025-09-23', '2025-09-24', '2025-09-25', '2025-09-26', '2025-09-28', '2025-09-29', '2025-09-30',
        '2025-12-07', '2025-12-10', '2025-12-11', '2025-12-12', '2025-12-14', '2025-12-15', '2025-12-16', '2025-12-17',
        '2025-12-18', '2025-12-19', '2025-12-21', '2025-12-22', '2025-12-23', '2025-12-24', '2025-12-25', '2025-12-26',
        '2025-12-28', '2025-12-29', '2025-12-30', '2025-12-31'
    ]
    bad_dates = pd.to_datetime(bad_dates)
    
    print(f"Original merged rows: {len(merged_df)}")
    merged_df = merged_df[~merged_df.index.normalize().isin(bad_dates)].copy()
    print(f"Rows after dropping unmatched roll dates: {len(merged_df)}")
    
    # 3. Calculate Target (Exhaustion)
    ret = merged_df['close'].diff()
    merged_df['vol_5s'] = ret.rolling(window=120, min_periods=30).std()
    
    k = 60 # 5 minutes
    
    R_k = 1.5 * merged_df['vol_5s'] * np.sqrt(k)
    fwd_min = merged_df['close'].shift(-k).rolling(window=k, min_periods=k).min()
    fwd_max = merged_df['close'].shift(-k).rolling(window=k, min_periods=k).max()
    prior_move = merged_df['close'] - merged_df['close'].shift(k)
    
    valid_idx = merged_df['close'].shift(-k).notna() & fwd_min.notna() & prior_move.notna()
    
    exh_mask = valid_idx & (prior_move.abs() > R_k)
    df_exh = merged_df[exh_mask].copy()
    
    prior_up = prior_move[exh_mask] > 0
    reversal_down = fwd_min[exh_mask] <= (df_exh['close'] - R_k[exh_mask])
    reversal_up = fwd_max[exh_mask] >= (df_exh['close'] + R_k[exh_mask])
    
    y_exh = np.where(prior_up, reversal_down, reversal_up).astype(int)
    print(f"Target prevalence: {y_exh.mean():.4f} (N={len(y_exh)})")
    
    # 4. Define the 3 Feature Layers
    non_feature_cols = ['cum_delta', 'divergence', 'delta', 'volume', 'open', 'high', 'low', 'close', 'vol_5s', 'quadrant']
    baseline_cols = [c for c in baseline_df.columns if c not in non_feature_cols and c in df_exh.columns]
    
    # Calculate Layer 2: Wick Absorption (FREE, OHLCV)
    total_range = (df_exh['high'] - df_exh['low']) + 1e-5
    df_exh['body_vol'] = (abs(df_exh['close'] - df_exh['open']) / total_range) * df_exh['volume']
    
    # Wick from close (absorption of the directional thrust)
    # If candle is UP (close > open), the wick from close is (high - close).
    # If candle is DN (close < open), the wick from close is (close - low).
    is_up = df_exh['close'] > df_exh['open']
    wick_close_len = np.where(is_up, df_exh['high'] - df_exh['close'], df_exh['close'] - df_exh['low'])
    df_exh['wick_close_vol'] = (wick_close_len / total_range) * df_exh['volume']
    
    layer_2_cols = baseline_cols + ['body_vol', 'wick_close_vol']
    
    # Calculate Layer 3: True Delta Absorption
    price_delta = df_exh['close'] - df_exh['open']
    df_exh['quadrant'] = 0
    df_exh.loc[(price_delta > 0) & (df_exh['delta'] > 0), 'quadrant'] = 1
    df_exh.loc[(price_delta > 0) & (df_exh['delta'] < 0), 'quadrant'] = 2
    df_exh.loc[(price_delta < 0) & (df_exh['delta'] < 0), 'quadrant'] = 3
    df_exh.loc[(price_delta < 0) & (df_exh['delta'] > 0), 'quadrant'] = 4
    
    # Delta path over the prior K move (approximation of intra-move path)
    df_exh['path_max_delta'] = merged_df['delta'].rolling(k).max()[exh_mask]
    df_exh['path_min_delta'] = merged_df['delta'].rolling(k).min()[exh_mask]
    
    layer_3_cols = layer_2_cols + ['cum_delta', 'divergence', 'delta', 'path_max_delta', 'path_min_delta', 'quadrant']
    delta_exclusive_cols = ['cum_delta', 'divergence', 'delta', 'path_max_delta', 'path_min_delta', 'quadrant']
    
    # 5. Temporal Split
    split_idx = int(len(df_exh) * 0.66)
    y_train = pd.Series(y_exh[:split_idx])
    y_test = pd.Series(y_exh[split_idx:])
    
    # Define models
    xgb_params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'n_jobs': -1, 'random_state': 42, 'tree_method': 'hist'}
    
    # Train L1 (Baseline)
    print("\n--- Training L1 (Baseline 416D) ---")
    m1 = XGBClassifier(**xgb_params).fit(df_exh[baseline_cols].iloc[:split_idx], y_train)
    auc_l1 = roc_auc_score(y_test, m1.predict_proba(df_exh[baseline_cols].iloc[split_idx:])[:, 1])
    print(f"L1 AUC: {auc_l1:.4f}")
    
    if auc_l1 < 0.52 or auc_l1 > 0.85:
        print("FAIL: L1 Baseline out of Plausibility Bounds [0.52, 0.85]. Stopping.")
        return
        
    # Train L2 (Wick Absorption)
    print("\n--- Training L2 (Free OHLCV Wicks) ---")
    m2 = XGBClassifier(**xgb_params).fit(df_exh[layer_2_cols].iloc[:split_idx], y_train)
    auc_l2 = roc_auc_score(y_test, m2.predict_proba(df_exh[layer_2_cols].iloc[split_idx:])[:, 1])
    print(f"L2 AUC: {auc_l2:.4f} (Lift over L1: {auc_l2 - auc_l1:+.4f})")
    
    # Train L3 (True Delta)
    print("\n--- Training L3 (True Delta Enriched) ---")
    m3 = XGBClassifier(**xgb_params).fit(df_exh[layer_3_cols].iloc[:split_idx], y_train)
    auc_l3 = roc_auc_score(y_test, m3.predict_proba(df_exh[layer_3_cols].iloc[split_idx:])[:, 1])
    print(f"L3 AUC: {auc_l3:.4f} (Lift over L2: {auc_l3 - auc_l2:+.4f})")
    
    # 6. Fourier Null on L3
    print("\n--- Fourier Null Gauntlet (20 seeds) on Layer 3 ---")
    surrogate_train = df_exh[layer_3_cols].iloc[:split_idx].copy()
    surrogate_test = df_exh[layer_3_cols].iloc[split_idx:].copy()
    
    null_aucs = []
    for seed in range(20):
        np.random.seed(seed)
        for c in delta_exclusive_cols:
            surrogate_train[c] = make_fourier_surrogate(df_exh[c].iloc[:split_idx])
            surrogate_test[c] = make_fourier_surrogate(df_exh[c].iloc[split_idx:])
            
        m_null = XGBClassifier(**xgb_params).fit(surrogate_train, y_train)
        null_auc = roc_auc_score(y_test, m_null.predict_proba(surrogate_test)[:, 1])
        null_aucs.append(null_auc)
        
    p95_null = np.percentile(null_aucs, 95)
    print(f"Null 95th Percentile AUC: {p95_null:.4f}")
    
    lift_l3_over_null = auc_l3 - p95_null
    
    # Verdict logic
    print("\n--- VERDICT ---")
    verdict = "BREAK"
    purchase = "NO (Stick to Free Wicks or Baseline)"
    
    if lift_l3_over_null > 0.005 and (auc_l3 - auc_l2) > 0.005:
        verdict = "MAKE"
        purchase = "YES (Buy Tick Data - L3 dominates)"
    elif (auc_l2 - auc_l1) > 0.005 and auc_l2 > p95_null:
        verdict = "MAKE"
        purchase = "NO (Free Wicks provide the edge)"
        
    print(f"Verdict: {verdict}")
    print(f"Tick Data Purchase: {purchase}")

    os.makedirs("research/order_flow_ablation/reports", exist_ok=True)
    with open("research/order_flow_ablation/reports/inflection_verdict.md", "w") as f:
        f.write("# Stage 1 Exhaustion Verdict\n\n")
        f.write("## Reconciliation Decomposition\n")
        f.write("- **Matched-Bar Ratio:** 1.0995 (10% mismatch localized heavily to roll-week dates).\n")
        f.write("- **Action Taken:** Roll-week dates were strictly excluded from the ablation to ensure L3 tests on clean, matched volume.\n\n")
        f.write("## 3-Layer Ablation Results\n")
        f.write(f"- **L1 Baseline AUC:** {auc_l1:.4f}\n")
        f.write(f"- **L2 Free Wicks AUC:** {auc_l2:.4f} (Lift over L1: {auc_l2 - auc_l1:+.4f})\n")
        f.write(f"- **L3 True Delta AUC:** {auc_l3:.4f} (Lift over L2: {auc_l3 - auc_l2:+.4f})\n\n")
        f.write("## Causal Gauntlet\n")
        f.write(f"- **L3 Fourier Null (95th pctile, N=20):** {p95_null:.4f}\n")
        f.write(f"- **L3 True Lift over Null:** {lift_l3_over_null:+.4f}\n\n")
        f.write(f"## Final Verdict: {verdict}\n")
        f.write(f"**Tick Data Purchase Decision:** {purchase}\n")

if __name__ == '__main__':
    run_stage_1_layered()
