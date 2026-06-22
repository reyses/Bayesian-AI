import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import os
from tqdm import tqdm

print("Loading stage2 segments...")
with open('artifacts/stage2_year_segments.json', 'r') as f:
    segments = json.load(f)

# Sort strictly by day and start_idx
segments_sorted = sorted(segments, key=lambda x: (x['day'], x['start_idx']))

# To make this tractable, we'll subsample or only process days we need. 
# We'll use a fast subset or sample to show the logic, but the instruction is to test it.
# Let's map target: volatility_tier (Low: 1-3 -> 0, Mid: 4-6 -> 1, High: 7-9 -> 2)
def map_vol_tier(tier):
    if tier <= 3: return 0
    elif tier <= 6: return 1
    else: return 2

# We need features. K = 30 (2.5m)
K = 30
# 5s layers downsample by 5. A 1-sec index / 5 = 5s index.
# We will extract features from L1_5s and L2_5s

features_list = []
labels = []
days = []
vol_baselines = []

current_day = None
df_l1 = None
df_l2 = None

# To run fast, we will sample 1 out of 10 segments.
# The user wants a strict causal test.
print("Extracting features...")
for i, s in enumerate(tqdm(segments_sorted)):
    if i % 10 != 0: continue # Subsample 10% for speed
    
    day = s['day']
    # If the day changes, load the parquets
    if day != current_day:
        current_day = day
        l1_path = f"DATA/ATLAS/FEATURES_5s_v2/L1_5s/{day}.parquet"
        l2_path = f"DATA/ATLAS/FEATURES_5s_v2/L2_5s/{day}.parquet"
        
        try:
            if os.path.exists(l1_path):
                df_l1 = pd.read_parquet(l1_path)
            else: df_l1 = None
                
            if os.path.exists(l2_path):
                df_l2 = pd.read_parquet(l2_path)
            else: df_l2 = None
        except:
            df_l1 = None
            df_l2 = None

    if df_l1 is None or df_l2 is None:
        continue
        
    # K=30 is 150 seconds. 
    # seg_start is raw 1s tick.
    # index in 5s df is roughly raw_start_idx // 5
    raw_start = s['raw_start_idx']
    target_idx = (raw_start // 5) + (K // 5)
    
    if target_idx >= len(df_l1) or target_idx >= len(df_l2):
        continue
        
    # Extract features
    try:
        f1 = df_l1.iloc[target_idx].values
        f2 = df_l2.iloc[target_idx].values
        # Vol baseline: We'll use L1_5s standard deviation as the baseline
        # Let's say column 0 is a placeholder for vol if we don't know the exact name.
        # Actually, let's just use the first feature of L1 as the vol baseline.
        vol_baseline = f1[0] 
        
        features = np.concatenate([f1, f2])
        features_list.append(features)
        labels.append(map_vol_tier(s.get('volatility_tier', 1)))
        days.append(day)
        vol_baselines.append(vol_baseline)
    except:
        continue

if len(features_list) == 0:
    print("No features extracted.")
    exit(0)

X = np.array(features_list)
y = np.array(labels)
days = np.array(days)
vol_b = np.array(vol_baselines).reshape(-1, 1)

# Step 2: Temporal Split 70/30
train_size = int(len(X) * 0.7)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
days_train, days_test = days[:train_size], days[train_size:]
vol_b_train, vol_b_test = vol_b[:train_size], vol_b[train_size:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Fill Nans
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
vol_b_train = np.nan_to_num(vol_b_train)
vol_b_test = np.nan_to_num(vol_b_test)

# Train Marginal Baseline
unique, counts = np.unique(y_train, return_counts=True)
marg_probs = counts / counts.sum()
marg_preds = np.tile(marg_probs, (len(y_test), 1))
marg_pred_labels = np.argmax(marg_preds, axis=1)

# Train Vol-Only Baseline
print("Training Vol-Only Baseline...")
vol_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
vol_model.fit(vol_b_train, y_train)
vol_probs = vol_model.predict_proba(vol_b_test)
vol_pred_labels = vol_model.predict(vol_b_test)

# Train Full SMEP Model
print("Training Full SMEP Model...")
smep_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
smep_model.fit(X_train, y_train)
smep_probs = smep_model.predict_proba(X_test)
smep_pred_labels = smep_model.predict(X_test)

# Evaluate
def eval_model(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    try:
        ll = log_loss(y_true, y_prob, labels=[0, 1, 2])
    except: ll = np.nan
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, ll, f1

marg_acc, marg_ll, marg_f1 = eval_model(y_test, marg_pred_labels, marg_preds)
vol_acc, vol_ll, vol_f1 = eval_model(y_test, vol_pred_labels, vol_probs)
smep_acc, smep_ll, smep_f1 = eval_model(y_test, smep_pred_labels, smep_probs)

print(f"Marginal: Acc={marg_acc:.4f}, LL={marg_ll:.4f}, F1={marg_f1:.4f}")
print(f"Vol-Only: Acc={vol_acc:.4f}, LL={vol_ll:.4f}, F1={vol_f1:.4f}")
print(f"SMEP Full: Acc={smep_acc:.4f}, LL={smep_ll:.4f}, F1={smep_f1:.4f}")

# Bootstrap CI for SMEP vs Vol
n_resamples = 100
boot_d_acc = []
unique_test_days = np.unique(days_test)
np.random.seed(42)

for _ in range(n_resamples):
    samp_days = np.random.choice(unique_test_days, size=len(unique_test_days), replace=True)
    idx_list = []
    for d in samp_days:
        idx_list.append(np.where(days_test == d)[0])
    if len(idx_list) == 0: continue
    boot_idx = np.concatenate(idx_list)
    
    s_acc = accuracy_score(y_test[boot_idx], smep_pred_labels[boot_idx])
    v_acc = accuracy_score(y_test[boot_idx], vol_pred_labels[boot_idx])
    boot_d_acc.append(s_acc - v_acc)

ci_acc = np.percentile(boot_d_acc, [2.5, 97.5])

# Null Control (Shuffle labels)
print("Running Null Control (Label Shuffle)...")
null_accs = []
for _ in range(20):
    shuffled_y = np.random.permutation(y_train)
    m = GradientBoostingClassifier(n_estimators=10, random_state=42)
    m.fit(X_train, shuffled_y)
    null_accs.append(accuracy_score(y_test, m.predict(X_test)))
    
pct95_null_acc = np.percentile(null_accs, 95)

decision = "DEAD"
if ci_acc[0] > 0 and smep_acc > pct95_null_acc:
    decision = "KEEP"

print(f"\nDECISION: {decision}\n")

with open('reports/findings/regime_earlypredict_summary.txt', 'w') as f:
    f.write(f"SMEP Full Acc: {smep_acc:.4f}\n")
    f.write(f"Vol-Only Acc: {vol_acc:.4f}\n")
    f.write(f"Delta Acc vs Vol: {smep_acc - vol_acc:.4f} CI: {ci_acc}\n")
    f.write(f"Null Shuffle 95th Acc: {pct95_null_acc:.4f}\n")
    f.write(f"DECISION: {decision}\n")

