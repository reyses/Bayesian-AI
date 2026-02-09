"""
Bayesian-AI - Result Inspection Tool
Quickly validates the trained probability table.
"""
import pickle
import sys
import os
import argparse

# Ensure we can import core modules if needed for unpickling custom classes
# (Though dicts usually don't need this, StateVector might if it's a key)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def inspect_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        # Handle if it's the wrapper class or just the table dict
        if hasattr(data, 'table'):
            table = data.table
        elif isinstance(data, dict) and 'table' in data:
            table = data['table']
        elif isinstance(data, dict):
            table = data # Assuming raw dict
        else:
            print("Unknown model format.")
            return

        total_states = len(table)
        print(f"Total states learned: {total_states}")

        # Identify high confidence states
        # Logic: Total samples >= 10 AND (Wins / (Total + 2)) >= 0.80
        high_prob_states = []
        for state, stats in table.items():
            # Support both object and dict access for stats if necessary
            # Assuming stats is a dict like {'wins': X, 'losses': Y, 'total': Z}
            if isinstance(stats, dict):
                wins = stats.get('wins', 0)
                total = stats.get('total', 0)
            else:
                # If stats is an object
                wins = getattr(stats, 'wins', 0)
                total = getattr(stats, 'total', 0)

            # Laplace smoothing denominator usually Total + 2 (for win/loss binary)
            # Probability = (Wins + 1) / (Total + 2)
            prob = (wins + 1) / (total + 2)

            if total >= 10 and prob >= 0.80:
                high_prob_states.append((state, prob, total))

        print(f"High-confidence states (80%+): {len(high_prob_states)}")

        if len(high_prob_states) > 0:
            print("\nTop 5 High-Confidence States:")
            # Sort by probability desc, then total desc
            high_prob_states.sort(key=lambda x: (x[1], x[2]), reverse=True)
            for i, (state, prob, total) in enumerate(high_prob_states[:5]):
                print(f"  {i+1}. Prob: {prob:.2%} | Samples: {total} | State: {state}")

    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Bayesian Probability Table")
    parser.add_argument("model_path", nargs='?', default="models/probability_table.pkl", help="Path to the .pkl file")

    args = parser.parse_args()
    inspect_model(args.model_path)
