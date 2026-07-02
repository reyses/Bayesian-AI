"""
Train a Logistic Regression classifier on the candidate primitives to act as a 
filter for cubic proposals.
Reads from DATA/cusp_picks/features/candidate_primitives.csv
Writes to DATA/cusp_picks/model.pkl
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

CSV_PATH = 'DATA/cusp_picks/features/candidate_primitives.csv'
MODEL_PATH = 'DATA/cusp_picks/model.pkl'

# The core causal features to train on. 
FEATURES = [
    'z_15s', 'z_1m', 'z_15m', 
    'slope_15s_3m', 'slope_15s_10m', 'slope_1m_10m',
    'slope_15m_5m', 'slope_15m_15m', 'slope_15m_decel',
    'curv_15m',
    'band_width', 'band_rank_60', 'sigma_15m_rank_60',
    'fan_width',
    # 'dist_15s_1m', 'dist_1m_15m', 'dist_15s_15m', (Removed, not present in compute_primitive_arrays)
    'align_up_count', 'align_down_count',
]

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Run extract_pick_primitives.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("Dataset is empty.")
        return

    print(f"Loaded {len(df)} candidates. Target=1 (accepted): {df['target'].sum()}")
    
    if df['target'].sum() < 10:
        print("Not enough accepted candidates to train a meaningful classifier yet. (Need >=10).")
        # We can still proceed, but it's risky
        if df['target'].sum() == 0:
            return

    # Drop rows with NaNs in features
    df_clean = df.dropna(subset=FEATURES).copy()
    print(f"Dropped {len(df) - len(df_clean)} rows with NaNs. {len(df_clean)} remain.")
    
    if len(df_clean) < 10:
        print("Too few rows remain after NaN drop.")
        return

    X = df_clean[FEATURES].values
    y = df_clean['target'].values

    # Stratified split to ensure we have positive examples in test set
    if df_clean['target'].sum() >= 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        # Fallback if too few positives
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test)

    # Train model
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    print("\n=== Model Evaluation ===")
    
    # Train metrics
    train_preds = clf.predict(X_train_scaled)
    train_probs = clf.predict_proba(X_train_scaled)[:, 1]
    train_auc = roc_auc_score(y_train, train_probs) if len(np.unique(y_train)) > 1 else np.nan
    print(f"Train AUC: {train_auc:.3f} | Accuracy: {accuracy_score(y_train, train_preds):.3f}")

    # Test metrics
    if len(X_test) > 0 and len(np.unique(y_test)) > 1:
        test_preds = clf.predict(X_test_scaled)
        test_probs = clf.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, test_probs)
        print(f"Test AUC:  {test_auc:.3f} | Accuracy: {accuracy_score(y_test, test_preds):.3f}")
        print("\nTest Classification Report:")
        print(classification_report(y_test, test_preds))
    else:
        print("Test set lacked multiple classes, skipping test metrics.")

    # Feature Importance (LR coefficients)
    print("\n=== Feature Importance (Logistic Regression Coefficients) ===")
    coefs = pd.Series(clf.coef_[0], index=FEATURES).sort_values(ascending=False)
    for feat, coef in coefs.items():
        print(f"  {feat:<20}: {coef:>6.3f}")

    # Save model and scaler
    model_data = {
        'model': clf,
        'scaler': scaler,
        'features': FEATURES
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"\nSaved model to {MODEL_PATH}")

if __name__ == '__main__':
    main()
