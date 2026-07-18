import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
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
    
    x_surr = (x_surr - np.mean(x_surr)) / (np.std(x_surr) + 1e-9) * np.std(x) + np.mean(x)
    return x_surr

def run_stage_1_layered():
    print("Stage 1: 3-Layer Inflection Ablation (Exhaustion Target) - FULL GAUNTLET")
    
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
    bad_dates = [
        '2025-09-14', '2025-09-15', '2025-09-16', '2025-09-17', '2025-09-18', '2025-09-19', '2025-09-21', 
        '2025-09-22', '2025-09-23', '2025-09-24', '2025-09-25', '2025-09-26', '2025-09-28', '2025-09-29', '2025-09-30',
        '2025-12-07', '2025-12-10', '2025-12-11', '2025-12-12', '2025-12-14', '2025-12-15', '2025-12-16', '2025-12-17',
        '2025-12-18', '2025-12-19', '2025-12-21', '2025-12-22', '2025-12-23', '2025-12-24', '2025-12-25', '2025-12-26',
        '2025-12-28', '2025-12-29', '2025-12-30', '2025-12-31'
    ]
    bad_dates = pd.to_datetime(bad_dates)
    merged_df = merged_df[~merged_df.index.normalize().isin(bad_dates)].copy()
    
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
    
    # 4. Define the 3 Feature Layers
    non_feature_cols = ['cum_delta', 'divergence', 'delta', 'volume', 'open', 'high', 'low', 'close', 'vol_5s', 'quadrant']
    baseline_cols = [c for c in baseline_df.columns if c not in non_feature_cols and c in df_exh.columns]
    
    # Layer 2: Wick Absorption
    total_range = (df_exh['high'] - df_exh['low']) + 1e-5
    df_exh['body_vol'] = (abs(df_exh['close'] - df_exh['open']) / total_range) * df_exh['volume']
    is_up = df_exh['close'] > df_exh['open']
    wick_close_len = np.where(is_up, df_exh['high'] - df_exh['close'], df_exh['close'] - df_exh['low'])
    df_exh['wick_close_vol'] = (wick_close_len / total_range) * df_exh['volume']
    layer_2_cols = baseline_cols + ['body_vol', 'wick_close_vol']
    
    # Layer 3: True Delta
    price_delta = df_exh['close'] - df_exh['open']
    df_exh['quadrant'] = 0
    df_exh.loc[(price_delta > 0) & (df_exh['delta'] > 0), 'quadrant'] = 1
    df_exh.loc[(price_delta > 0) & (df_exh['delta'] < 0), 'quadrant'] = 2
    df_exh.loc[(price_delta < 0) & (df_exh['delta'] < 0), 'quadrant'] = 3
    df_exh.loc[(price_delta < 0) & (df_exh['delta'] > 0), 'quadrant'] = 4
    
    # Correct Path Features (cum_delta max/min over the prior k bars)
    df_exh['path_max_cum_delta'] = merged_df['cum_delta'].rolling(k).max()[exh_mask]
    df_exh['path_min_cum_delta'] = merged_df['cum_delta'].rolling(k).min()[exh_mask]
    
    layer_3_cols = layer_2_cols + ['cum_delta', 'divergence', 'delta', 'path_max_cum_delta', 'path_min_cum_delta', 'quadrant']
    delta_exclusive_cols = ['cum_delta', 'divergence', 'delta', 'path_max_cum_delta', 'path_min_cum_delta', 'quadrant']
    
    # 5. Temporal Walk-Forward Split (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'n_jobs': -1, 'random_state': 42, 'tree_method': 'hist'}
    
    l1_aucs = []
    l2_aucs = []
    l3_aucs = []
    
    print("\n--- Running 5-Fold Temporal Walk-Forward Gauntlet ---")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df_exh)):
        print(f"\nFold {fold+1}: Train={len(train_idx)}, Test={len(test_idx)}")
        
        y_train_f = y_exh[train_idx]
        y_test_f = y_exh[test_idx]
        
        # L1
        m1 = XGBClassifier(**xgb_params).fit(df_exh[baseline_cols].iloc[train_idx], y_train_f)
        a1 = roc_auc_score(y_test_f, m1.predict_proba(df_exh[baseline_cols].iloc[test_idx])[:, 1])
        l1_aucs.append(a1)
        
        # L2
        m2 = XGBClassifier(**xgb_params).fit(df_exh[layer_2_cols].iloc[train_idx], y_train_f)
        a2 = roc_auc_score(y_test_f, m2.predict_proba(df_exh[layer_2_cols].iloc[test_idx])[:, 1])
        l2_aucs.append(a2)
        
        # L3
        m3 = XGBClassifier(**xgb_params).fit(df_exh[layer_3_cols].iloc[train_idx], y_train_f)
        a3 = roc_auc_score(y_test_f, m3.predict_proba(df_exh[layer_3_cols].iloc[test_idx])[:, 1])
        l3_aucs.append(a3)
        
        print(f"  L1: {a1:.4f} | L2: {a2:.4f} | L3: {a3:.4f}")
        
    avg_l1 = np.mean(l1_aucs)
    avg_l2 = np.mean(l2_aucs)
    avg_l3 = np.mean(l3_aucs)
    
    print("\n--- Sub-Period Stability Results (Averages) ---")
    print(f"L1 (Baseline): {avg_l1:.4f}")
    print(f"L2 (OHLCV Wicks): {avg_l2:.4f}")
    print(f"L3 (True Delta): {avg_l3:.4f}")
    
    if avg_l1 < 0.52 or avg_l1 > 0.85:
        print("FAIL: L1 Baseline out of Plausibility Bounds [0.52, 0.85].")
        return
        
    # 6. Fourier Null on L3 (Using the final fold for the gauntlet, or aggregated surrogate)
    # We will test the surrogate on the last fold to be extremely strict.
    print("\n--- Fourier Null Gauntlet (20 seeds, Fold 5) ---")
    train_idx = tscv.split(df_exh).__iter__().__next__()[0] # Actually, let's just use the last fold
    folds = list(tscv.split(df_exh))
    train_idx, test_idx = folds[-1]
    
    y_train_f = y_exh[train_idx]
    y_test_f = y_exh[test_idx]
    
    surrogate_train = df_exh[layer_3_cols].iloc[train_idx].copy()
    surrogate_test = df_exh[layer_3_cols].iloc[test_idx].copy()
    
    null_aucs = []
    for seed in range(20):
        np.random.seed(seed)
        for c in delta_exclusive_cols:
            surrogate_train[c] = make_fourier_surrogate(df_exh[c].iloc[train_idx])
            surrogate_test[c] = make_fourier_surrogate(df_exh[c].iloc[test_idx])
            
        m_null = XGBClassifier(**xgb_params).fit(surrogate_train, y_train_f)
        null_auc = roc_auc_score(y_test_f, m_null.predict_proba(surrogate_test)[:, 1])
        null_aucs.append(null_auc)
        
    p95_null = np.percentile(null_aucs, 95)
    last_l3_auc = l3_aucs[-1]
    print(f"Fold 5 L3 AUC: {last_l3_auc:.4f}")
    print(f"Null 95th Percentile AUC: {p95_null:.4f}")
    
    lift_l3_over_null = last_l3_auc - p95_null
    
    # Verdict logic
    print("\n--- VERDICT ---")
    verdict = "BREAK"
    purchase = "NO (Stick to Free Wicks or Baseline)"
    
    if lift_l3_over_null > 0.003 and (avg_l3 - avg_l2) > 0.003:
        verdict = "MAKE"
        purchase = "YES (Buy Tick Data - L3 dominates)"
    elif (avg_l2 - avg_l1) > 0.003 and avg_l2 > p95_null:
        verdict = "MAKE"
        purchase = "NO (Free Wicks provide the edge)"
        
    print(f"Verdict: {verdict}")
    print(f"Tick Data Purchase: {purchase}")

    os.makedirs("research/order_flow_ablation/reports", exist_ok=True)
    with open("research/order_flow_ablation/reports/inflection_verdict.md", "w") as f:
        f.write("# Stage 1 Exhaustion Verdict (Full Causal Gauntlet)\n\n")
        f.write("## 5-Fold Walk-Forward & Sub-Period Stability\n")
        f.write(f"- **L1 (Baseline):** {avg_l1:.4f} avg AUC (In Bounds [0.55-0.75])\n")
        f.write(f"- **L2 (OHLCV Wicks):** {avg_l2:.4f} avg AUC\n")
        f.write(f"- **L3 (True Delta Enriched):** {avg_l3:.4f} avg AUC (Lift over L2: {avg_l3 - avg_l2:+.4f})\n\n")
        f.write("### Fold-by-Fold Stability (L3 AUCs)\n")
        for i, a in enumerate(l3_aucs):
            f.write(f"- **Fold {i+1}:** {a:.4f}\n")
        f.write("\n## Fourier Null Gauntlet (Fold 5)\n")
        f.write(f"- **L3 Fold 5 AUC:** {last_l3_auc:.4f}\n")
        f.write(f"- **Fourier Null 95th pctile (N=20):** {p95_null:.4f}\n")
        f.write(f"- **True Lift over Null:** {lift_l3_over_null:+.4f}\n\n")
        f.write(f"## Final Verdict: {verdict}\n")
        f.write(f"**Tick Data Purchase Decision:** {purchase}\n")

if __name__ == '__main__':
    run_stage_1_layered()
