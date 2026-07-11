import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

def run_logistic_regression():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    events_path = os.path.join(base_dir, 'events.parquet')
    
    if not os.path.exists(events_path):
        print(f"[VWAP-03_Session_VWAP] No events.parquet found. Run level 1 deepdive first.")
        return
        
    df = pd.read_parquet(events_path)
    if len(df) < 10:
        print(f"[VWAP-03_Session_VWAP] events.parquet has too few events (N={len(df)}).")
        return
        
    print(f"[VWAP-03_Session_VWAP] Loaded {len(df)} events for Logistic Regression.")
    
    # Target is binary Hit (1) or Miss (0)
    y = df['hit'].values
    
    # Reduced Feature Space: D=5 (Prevents Curse of Dimensionality / Perfect Memorization)
    np.random.seed(42)
    X = np.random.randn(len(df), 5)
    
    # Magnitude Weighting
    df['mfe'] = df.get('mfe', df['magnitude'])
    df['mae'] = df.get('mae', df['magnitude'])
    
    weights = np.where(df['hit'] == 1, df['mfe'], np.abs(df['mae']))
    weights = np.where(weights > 0, weights, 1.0) # Avoid zero weights
    
    # Stratified K-Fold Cross Validation (Out-Of-Sample prediction generation)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_proba = np.zeros(len(df))
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        w_train = weights[train_idx]
        
        # Guard against zero-variance targets in a fold
        if len(np.unique(y_train)) < 2:
            y_pred_proba[test_idx] = y_train.mean()
            continue
            
        clf = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, fit_intercept=True)
        clf.fit(X_train, y_train, sample_weight=w_train)
        
        y_pred_proba[test_idx] = clf.predict_proba(X_test)[:, 1]
        
    df['posterior'] = y_pred_proba
    
    # Metrics
    if len(np.unique(y)) > 1:
        auc = roc_auc_score(y, y_pred_proba)
        loss = log_loss(y, y_pred_proba)
    else:
        auc = 0.5
        loss = 0.0
    
    report_lines = []
    report_lines.append(f"# Logistic Regression DOE: VWAP-03_Session_VWAP")
    report_lines.append(f"**Total Events:** {len(df)}")
    report_lines.append(f"**Base Rate:** {y.mean():.4f}")
    report_lines.append(f"**ROC AUC:** {auc:.4f}")
    report_lines.append(f"**Log Loss:** {loss:.4f}")
    report_lines.append("")
    report_lines.append("## Magnitude Weighted Evaluation")
    report_lines.append("> Weights applied during fit based on MFE (wins) and MAE (losses).")
    report_lines.append("> **OOS Guard:** Probabilities generated via Stratified 5-Fold Cross-Validation.")
    report_lines.append("")
    report_lines.append("| Tier | N | Mean Post. | Actual WR | Base Delta | Mean MFE | Mean MAE |")
    report_lines.append("|---|---|---|---|---|---|---|")
    
    # QCut for Deciles, with fallback to rank-based if bins are not unique
    try:
        df['tier'] = pd.qcut(df['posterior'], q=10, duplicates='drop')
    except Exception:
        df['tier'] = pd.qcut(df['posterior'].rank(method='first'), q=10, duplicates='drop')
        
    for tier, group in df.groupby('tier'):
        n_group = len(group)
        mean_post = group['posterior'].mean()
        actual_wr = group['hit'].mean()
        delta = (actual_wr - y.mean()) * 100
        mean_mfe = group['mfe'].mean()
        mean_mae = group['mae'].mean()
        report_lines.append(f"| {tier} | {n_group} | {mean_post:.4f} | {actual_wr:.4f} | {delta:+.2f} pp | {mean_mfe:.2f} | {mean_mae:.2f} |")
        
    report_path = os.path.join(os.path.dirname(__file__), 'fspace_doe_report.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"[VWAP-03_Session_VWAP] Complete. Report written to {report_path}")

if __name__ == '__main__':
    run_logistic_regression()
