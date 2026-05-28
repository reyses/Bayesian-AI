import json
import os

CONFIG_FILE = 'curriculum_params.json'

def init_default_config():
    """Initializes the baseline hyperparameter tracking JSON."""
    default_config = {
        "learning_rate": 0.005,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.05,
        "gamma": 0.99,
        "vtrace_clip_rho": 1.0,
        "vtrace_clip_c": 1.0,
        "eval_thresholds": {
            "min_metric_n": 0.0,
            "min_auc": 0.5,
            "min_pnl_mode_ci_lower": 0.0
        },
        "history": [] # Stores historical metrics for each segment
    }
    
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"[INFO] Created new configuration file: {CONFIG_FILE}")

def load_config():
    """Loads the live hyperparameter config."""
    if not os.path.exists(CONFIG_FILE):
        init_default_config()
        
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def save_segment_metrics(segment_name, metrics, passed):
    """Appends the evaluation results of a Walk-Forward segment into the JSON history."""
    config = load_config()
    
    record = {
        "segment": segment_name,
        "passed": passed,
        "metrics": metrics
    }
    
    config["history"].append(record)
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"[INFO] Segment {segment_name} metrics appended to {CONFIG_FILE}")

if __name__ == "__main__":
    init_default_config()
