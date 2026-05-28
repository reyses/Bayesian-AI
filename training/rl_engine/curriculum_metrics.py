import numpy as np
from scipy import stats
from sklearn import metrics

def compute_mode_ci(data_array, confidence=0.95, num_bootstraps=1000):
    """
    Computes the Mode of the distribution (the 'bulk' of trades) 
    and returns a bootstrapped Confidence Interval for the Mode.
    """
    if len(data_array) < 10:
        return 0.0, (0.0, 0.0)
        
    modes = []
    for _ in range(num_bootstraps):
        sample = np.random.choice(data_array, size=len(data_array), replace=True)
        # Using a histogram-based mode estimation for speed
        counts, bins = np.histogram(sample, bins=30)
        max_bin_idx = np.argmax(counts)
        mode_val = (bins[max_bin_idx] + bins[max_bin_idx+1]) / 2.0
        modes.append(mode_val)
        
    modes = np.array(modes)
    lower_bound = np.percentile(modes, (1 - confidence) / 2.0 * 100)
    upper_bound = np.percentile(modes, (1 + confidence) / 2.0 * 100)
    
    # Original data mode
    counts, bins = np.histogram(data_array, bins=30)
    max_bin_idx = np.argmax(counts)
    actual_mode = (bins[max_bin_idx] + bins[max_bin_idx+1]) / 2.0
    
    return actual_mode, (lower_bound, upper_bound)

def evaluate_curriculum_segment(trade_pnls, trade_durations, config=None):
    """
    Evaluates the 4 strict metrics for the Curriculum Progression Gate:
    1. Metric (n)
    2. Confidence Interval of Mode (Duration)
    3. Confidence Interval of Mode (PNL)
    4. AUC
    """
    if len(trade_pnls) == 0:
        return False, {}
        
    trade_pnls = np.array(trade_pnls)
    trade_durations = np.array(trade_durations)
    
    # 1. Metric (n)
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls <= 0]
    mean_win = np.mean(wins) if len(wins) > 0 else 0.0
    mean_loss = abs(np.mean(losses)) if len(losses) > 0 else 1.0
    metric_n = (mean_win / mean_loss) - 1.0
    
    # 2. Mode CI (Duration)
    dur_mode, (dur_lower, dur_upper) = compute_mode_ci(trade_durations)
    
    # 3. Mode CI (PNL)
    pnl_mode, (pnl_lower, pnl_upper) = compute_mode_ci(trade_pnls)
    
    # 4. AUC
    cumulative_pnl = np.cumsum(trade_pnls)
    normalized_time = np.linspace(0, 1, len(cumulative_pnl))
    normalized_pnl = (cumulative_pnl - np.min(cumulative_pnl)) / (np.max(cumulative_pnl) - np.min(cumulative_pnl) + 1e-9)
    auc_score = metrics.auc(normalized_time, normalized_pnl)
    
    if config and "eval_thresholds" in config:
        min_n = config["eval_thresholds"].get("min_metric_n", 0.0)
        min_auc = config["eval_thresholds"].get("min_auc", 0.5)
        min_pnl = config["eval_thresholds"].get("min_pnl_mode_ci_lower", 0.0)
    else:
        min_n = 0.0
        min_auc = 0.5
        min_pnl = 0.0
        
    passed = bool(metric_n > min_n and pnl_lower > min_pnl and auc_score > min_auc)
    
    results = {
        'metric_n': metric_n,
        'dur_mode_ci': (dur_lower, dur_upper),
        'pnl_mode_ci': (pnl_lower, pnl_upper),
        'auc': auc_score,
        'passed': passed
    }
    
    return passed, results

if __name__ == "__main__":
    # Test stub
    dummy_pnls = np.random.normal(loc=0.5, scale=1.0, size=100)
    dummy_durs = np.random.randint(5, 50, size=100)
    passed, metrics = evaluate_curriculum_segment(dummy_pnls, dummy_durs)
    print(f"Passed: {passed} | Metrics: {metrics}")
