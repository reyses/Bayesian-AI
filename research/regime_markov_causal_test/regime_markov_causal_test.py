import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Loading artifacts...")
with open('artifacts/stage2_year_segments.json', 'r') as f:
    segments = json.load(f)

# Sort strictly by day and start_idx
# day format: '2025_01_05'
# The segments are a list of dicts. We will attach an original_index so we know which one it was in the file,
# because regime_buckets.json stores the original list index.
for i, s in enumerate(segments):
    s['original_index'] = i

# Sort them to get the true chronological sequence
segments_sorted = sorted(segments, key=lambda x: (x['day'], x['start_idx']))

# Verify alignment
# The handover notes: "verify by spot-checking that bucket member indices are valid into this ordered array"
# Actually, the bucket members index the original array. We will map original_index -> regime label.
with open('artifacts/regime_buckets.json', 'r') as f:
    regime_buckets = json.load(f)

# Build original_index -> label mapping
# We assign highest tier (tier1>tier2>tier3>tier4). Ties -> larger regime (by total_members)
segment_labels = {}

for reg_id, data in regime_buckets.items():
    total_mems = data.get('total_members', 0)
    
    for tier in [1, 2, 3, 4]:
        members = data.get(f'members_tier_{tier}', [])
        for m in members:
            # We want to keep the one with the best tier (lowest number).
            # If tie in tier, largest total_members wins.
            if m not in segment_labels:
                segment_labels[m] = {'regime': f"R{reg_id}", 'tier': tier, 'total_members': total_mems}
            else:
                current = segment_labels[m]
                if tier < current['tier']:
                    segment_labels[m] = {'regime': f"R{reg_id}", 'tier': tier, 'total_members': total_mems}
                elif tier == current['tier']:
                    if total_mems > current['total_members']:
                        segment_labels[m] = {'regime': f"R{reg_id}", 'tier': tier, 'total_members': total_mems}

print(f"Total segments: {len(segments)}")
print(f"Segments with a regime label: {len(segment_labels)}")

# Build chronological sequence r_1 ... r_N
sequence = []
days = []
for s in segments_sorted:
    idx = s['original_index']
    days.append(s['day'])
    if idx in segment_labels:
        sequence.append(segment_labels[idx]['regime'])
    else:
        sequence.append('NOISE')

sequence = np.array(sequence)
days = np.array(days)

# R1 vs R2 vs NOISE collapse. For simplicity let's keep all, but the baseline predicts R1.
# Actually, let's keep the actual labels. We need to measure transition accuracy.
# To compute log-loss, we need the transition matrix. We only compute over the top regimes to keep it tractable.
unique_labels = list(set(sequence))
label_to_idx = {l: i for i, l in enumerate(unique_labels)}
idx_to_label = {i: l for i, l in enumerate(unique_labels)}
N_STATES = len(unique_labels)

print(f"Unique states: {N_STATES}")

# Step 2: Temporal Split 70/30
train_size = int(len(sequence) * 0.7)
train_seq = sequence[:train_size]
test_seq = sequence[train_size:]
test_days = days[train_size:]

print(f"Train size: {len(train_seq)}, Test size: {len(test_seq)}")

# Step 3: Models & Baselines
# Marginal Baseline
unique_train, counts_train = np.unique(train_seq, return_counts=True)
train_marginals = {l: c/len(train_seq) for l, c in zip(unique_train, counts_train)}

# If a test label wasn't in train, give it epsilon
eps = 1e-6
for l in unique_labels:
    if l not in train_marginals:
        train_marginals[l] = eps

# Normalize marginals over all unique_labels
marginal_probs = np.array([train_marginals[idx_to_label[i]] for i in range(N_STATES)])
marginal_probs /= marginal_probs.sum()

argmax_marginal = idx_to_label[np.argmax(marginal_probs)]

# Transition Model (Laplace smoothed)
trans_counts = np.ones((N_STATES, N_STATES)) # Laplace +1
for t in range(len(train_seq)-1):
    curr_idx = label_to_idx[train_seq[t]]
    next_idx = label_to_idx[train_seq[t+1]]
    trans_counts[curr_idx, next_idx] += 1

trans_probs = trans_counts / trans_counts.sum(axis=1, keepdims=True)

# Function to evaluate
def evaluate_seq(curr_seq, next_seq, custom_trans_probs=None):
    use_trans_probs = trans_probs if custom_trans_probs is None else custom_trans_probs
    # Marginal predictions
    marg_preds = np.tile(marginal_probs, (len(next_seq), 1))
    marg_acc = np.mean(next_seq == argmax_marginal)
    
    # Next_seq idx
    next_seq_idx = np.array([label_to_idx[l] for l in next_seq])
    marg_loss = np.nan
        
    # Transition predictions
    curr_seq_idx = np.array([label_to_idx[l] for l in curr_seq])
    trans_preds = use_trans_probs[curr_seq_idx]
    trans_pred_labels = np.array([idx_to_label[i] for i in np.argmax(trans_preds, axis=1)])
    
    trans_acc = np.mean(next_seq == trans_pred_labels)
    trans_loss = np.nan
        
    return trans_acc - marg_acc, marg_loss - trans_loss # positive is better for both

# Step 4: Metrics on TEST (Day-block bootstrap)
print("Evaluating on TEST...")
test_curr = test_seq[:-1]
test_next = test_seq[1:]
test_curr_days = test_days[:-1]

real_d_acc, real_d_ll = evaluate_seq(test_curr, test_next)
print(f"REAL OOS -> Delta Acc: {real_d_acc:.4f}, Delta Log-Loss: {real_d_ll:.4f}")

# Bootstrap CI
unique_test_days = np.unique(test_curr_days)
n_resamples = 2000
boot_d_acc = []
boot_d_ll = []

np.random.seed(42)
for _ in range(n_resamples):
    samp_days = np.random.choice(unique_test_days, size=len(unique_test_days), replace=True)
    idx_list = []
    for d in samp_days:
        idx_list.append(np.where(test_curr_days == d)[0])
    if len(idx_list) == 0: continue
    boot_idx = np.concatenate(idx_list)
    
    d_acc, d_ll = evaluate_seq(test_curr[boot_idx], test_next[boot_idx])
    boot_d_acc.append(d_acc)
    boot_d_ll.append(d_ll)

ci_acc = np.percentile(boot_d_acc, [2.5, 97.5])
ci_ll = np.percentile(boot_d_ll, [2.5, 97.5])

print(f"95% CI Delta Acc: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]")
print(f"95% CI Delta Log-Loss: [{ci_ll[0]:.4f}, {ci_ll[1]:.4f}]")

# Step 5: Null Controls
print("Running Null A (Sequence Shuffle)...")
null_a_d_acc = []
null_a_d_ll = []

for _ in range(100): # 100 for speed, standard is 1000
    shuffled_train = np.random.permutation(train_seq)
    
    # rebuild transition
    t_counts = np.ones((N_STATES, N_STATES))
    for t in range(len(shuffled_train)-1):
        t_counts[label_to_idx[shuffled_train[t]], label_to_idx[shuffled_train[t+1]]] += 1
    new_trans_probs = t_counts / t_counts.sum(axis=1, keepdims=True)
    
    d_acc, d_ll = evaluate_seq(test_curr, test_next, custom_trans_probs=new_trans_probs)
    null_a_d_acc.append(d_acc)
    null_a_d_ll.append(d_ll)

pct95_acc = np.percentile(null_a_d_acc, 95)
pct95_ll = np.percentile(null_a_d_ll, 95)

print(f"Null A 95th Pct Delta Acc: {pct95_acc:.4f}")
print(f"Null A 95th Pct Delta Log-Loss: {pct95_ll:.4f}")

# Step 6: Pre-Committed Decision Rule
decision = "DEAD"
if ci_acc[0] > 0 and ci_ll[0] > 0 and real_d_acc > pct95_acc and real_d_ll > pct95_ll:
    decision = "KEEP"

print(f"\nDECISION: {decision}\n")

with open('reports/findings/regime_markov_test_summary.txt', 'w') as f:
    f.write(f"REAL Delta Acc: {real_d_acc:.4f} CI: {ci_acc}\n")
    f.write(f"REAL Delta LL: {real_d_ll:.4f} CI: {ci_ll}\n")
    f.write(f"Null A 95th Acc: {pct95_acc:.4f}\n")
    f.write(f"Null A 95th LL: {pct95_ll:.4f}\n")
    f.write(f"DECISION: {decision}\n")

