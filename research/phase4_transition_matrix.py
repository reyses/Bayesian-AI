import json
import numpy as np

def build_transition_matrix():
    print("[Markov] Loading Regime Buckets...")
    with open('artifacts/regime_buckets.json', 'r') as f:
        buckets = json.load(f)
        
    N_segments = 80717
    # Initialize all segments to 0 (NOISE)
    state_timeline = np.zeros(N_segments, dtype=np.int32)
    
    print("[Markov] Reconstructing chronological timeline...")
    # Invert mapping: map every segment index back to its parent Regime ID
    for b_id, data in buckets.items():
        regime_id = int(b_id)
        # We include all 4 tiers of members to get the full structural map
        members = data['members_tier_1'] + data['members_tier_2'] + data['members_tier_3'] + data['members_tier_4']
        for m in members:
            state_timeline[m] = regime_id
            
    print("[Markov] Building the Massive Transition Probability Matrix...")
    num_regimes = len(buckets) + 1 # Include 0 for NOISE
    transition_counts = np.zeros((num_regimes, num_regimes), dtype=np.int64)
    
    # Chronological sweep: record every physical transition in the MNQ
    for i in range(N_segments - 1):
        current_state = state_timeline[i]
        next_state = state_timeline[i+1]
        transition_counts[current_state, next_state] += 1
        
    # Analyze transitions for the top 5 dominant regimes
    print("\n--- 'IF THIS, THEN THAT' (Top 5 Dominant Regimes) ---")
    for i in range(1, 6):
        total_transitions_out = np.sum(transition_counts[i])
        if total_transitions_out == 0: continue
        
        # Get the row of transition counts from Regime i
        row = transition_counts[i]
        
        # Sort the next states by most probable
        top_destinations = np.argsort(row)[::-1]
        
        print(f"\nIf CURRENT STATE == Regime {i} (Evaluated {total_transitions_out} physical transitions):")
        for j in range(5):
            dest_regime = top_destinations[j]
            count = row[dest_regime]
            prob = (count / total_transitions_out) * 100
            
            # Formatting
            if dest_regime == 0:
                dest_name = "NOISE (Chaos/Unclassified)"
            elif dest_regime == i:
                dest_name = f"STAY in Regime {i}"
            else:
                dest_name = f"Regime {dest_regime}"
                
            print(f"  -> {prob:>5.2f}% chance next state is: {dest_name:<28} ({count} times)")
            
    # Also analyze NOISE
    print("\n--- IF CURRENT STATE == NOISE (Chaos) ---")
    row = transition_counts[0]
    total_out = np.sum(row)
    top_destinations = np.argsort(row)[::-1]
    print(f"When the market devolves into Noise (Evaluated {total_out} times):")
    for j in range(5):
        dest_regime = top_destinations[j]
        count = row[dest_regime]
        prob = (count / total_out) * 100
        
        if dest_regime == 0:
            dest_name = "STAY in NOISE"
        else:
            dest_name = f"Emerge into Regime {dest_regime}"
            
        print(f"  -> {prob:>5.2f}% chance next state is: {dest_name:<28} ({count} times)")

    # Save to disk
    np.save('artifacts/transition_matrix.npy', transition_counts)
    print(f"\n[Markov] Saved full {num_regimes}x{num_regimes} matrix to artifacts/transition_matrix.npy")
    
if __name__ == '__main__':
    build_transition_matrix()
