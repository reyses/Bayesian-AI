import numpy as np
from scipy import stats

def compute_mode_ci(data_array, confidence=0.95, num_bootstraps=1000):
    if len(data_array) < 10:
        return 0.0, (0.0, 0.0)
        
    modes = []
    for _ in range(num_bootstraps):
        sample = np.random.choice(data_array, size=len(data_array), replace=True)
        counts, bins = np.histogram(sample, bins=30)
        max_bin_idx = np.argmax(counts)
        modes.append((bins[max_bin_idx] + bins[max_bin_idx+1]) / 2.0)
        
    modes = np.array(modes)
    lower = np.percentile(modes, (1 - confidence) / 2.0 * 100)
    upper = np.percentile(modes, (1 + confidence) / 2.0 * 100)
    
    counts, bins = np.histogram(data_array, bins=30)
    max_bin_idx = np.argmax(counts)
    actual_mode = (bins[max_bin_idx] + bins[max_bin_idx+1]) / 2.0
    return actual_mode, (lower, upper)

def compute_mean_ci(data_array, confidence=0.95, num_bootstraps=2000):
    if len(data_array) < 10:
        return 0.0, (0.0, 0.0)
    
    means = []
    for _ in range(num_bootstraps):
        sample = np.random.choice(data_array, size=len(data_array), replace=True)
        means.append(np.mean(sample))
        
    means = np.array(means)
    lower = np.percentile(means, (1 - confidence) / 2.0 * 100)
    upper = np.percentile(means, (1 + confidence) / 2.0 * 100)
    return np.mean(data_array), (lower, upper)

def calculate_max_drawdown(trade_pnls):
    if len(trade_pnls) == 0:
        return 0.0
    cumulative = np.cumsum(trade_pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    return np.max(drawdowns)

def evaluate_is_metrics(trade_pnls, trade_durations):
    """
    Computes basic trajectory metrics for In-Sample monitoring.
    AUC is strictly removed from all logic.
    """
    if len(trade_pnls) == 0:
        return {'metric_n': 0.0, 'total_pnl': 0.0, 'pnl_mode_ci': (0.0,0.0), 'dur_mode_ci': (0.0,0.0), 'avg_loss': 0.0}
        
    trade_pnls = np.array(trade_pnls)
    trade_durations = np.array(trade_durations)
    
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls <= 0]
    gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
    gross_loss = np.sum(np.abs(losses)) if len(losses) > 0 else 1e-9
    metric_n = (gross_profit / gross_loss) - 1.0
    
    _, pnl_ci = compute_mode_ci(trade_pnls)
    _, dur_ci = compute_mode_ci(trade_durations)
    
    return {
        'metric_n': metric_n,
        'total_pnl': np.sum(trade_pnls),
        'pnl_mode_ci': pnl_ci,
        'dur_mode_ci': dur_ci,
        'trade_count': len(trade_pnls)
    }

class OOSDiagnosticsSuite:
    def __init__(self):
        self.all_pnls = []
        self.all_durs = []
        self.all_mfe_avail = []
        self.all_mfe_trade = []
        self.all_mae = []
        
    def add_segment_data(self, pnls, durs, mfe_avail, mfe_trade, mae):
        self.all_pnls.extend(pnls)
        self.all_durs.extend(durs)
        self.all_mfe_avail.extend(mfe_avail)
        self.all_mfe_trade.extend(mfe_trade)
        self.all_mae.extend(mae)
        
        # Compute and return Segment-Level Diagnostics
        seg_pnls = np.array(pnls)
        seg_mfe_avail = np.array(mfe_avail)
        seg_mfe_trade = np.array(mfe_trade)
        
        if len(seg_pnls) == 0:
            return {}
            
        _, pnl_mode_ci = compute_mode_ci(seg_pnls)
        _, dur_mode_ci = compute_mode_ci(np.array(durs))
        
        cap_avail = np.sum(seg_pnls) / (np.sum(seg_mfe_avail) + 1e-9)
        cap_trade = np.sum(seg_pnls) / (np.sum(seg_mfe_trade) + 1e-9)
        
        return {
            'trade_count': len(pnls),
            'pnl_mode_ci': pnl_mode_ci,
            'dur_mode_ci': dur_mode_ci,
            'max_drawdown': calculate_max_drawdown(seg_pnls),
            'cap_vs_avail': cap_avail,
            'cap_vs_trade': cap_trade,
            'avg_mae': np.mean(mae) if len(mae) > 0 else 0.0
        }
        
    def get_pooled_diagnostics(self):
        """
        Computes the grand pooled aggregate across all OOS trades.
        None of these numbers influence weights, control flow, or penalties.
        """
        pnls = np.array(self.all_pnls)
        if len(pnls) == 0:
            return {}
            
        mean_pnl, mean_ci = compute_mean_ci(pnls)
        
        cap_avail = np.sum(pnls) / (np.sum(self.all_mfe_avail) + 1e-9)
        cap_trade = np.sum(pnls) / (np.sum(self.all_mfe_trade) + 1e-9)
        
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = np.sum(np.abs(losses)) if len(losses) > 0 else 1e-9
        profit_factor = gross_profit / gross_loss
        
        return {
            'total_trades': len(pnls),
            'pooled_mean_pnl': mean_pnl,
            'pooled_mean_pnl_ci': mean_ci,
            'pooled_max_drawdown': calculate_max_drawdown(pnls),
            'pooled_cap_vs_avail': cap_avail,
            'pooled_cap_vs_trade': cap_trade,
            'pooled_avg_mae': np.mean(self.all_mae),
            'profit_factor': profit_factor,
            'total_net_pnl': np.sum(pnls)
        }
