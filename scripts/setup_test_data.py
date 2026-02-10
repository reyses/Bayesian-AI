"""
Setup Test Data Script
Creates necessary directories for the CI/CD pipeline.
"""
import os
import sys

def main():
    print("Setting up test data directories...")

    # Create required directories
    directories = [
        "data/raw",
        "models",
        "debug_outputs",
        "checkpoints"
    ]

    for d in directories:
        os.makedirs(d, exist_ok=True)
        print(f"Verified directory: {d}")

    # Create a dummy probability table if it doesn't exist (to prevent runtime errors)
    # Some tests might check for this file
    prob_table_path = "models/probability_table.pkl"
    if not os.path.exists(prob_table_path):
        import pickle
        with open(prob_table_path, 'wb') as f:
            pickle.dump({}, f)
        print(f"Created dummy {prob_table_path}")

    print("Setup complete.")

if __name__ == "__main__":
    main()
