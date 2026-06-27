import matplotlib.pyplot as plt
import numpy as np
import datetime
import pytz
import os
import csv

def plot_epoch_summary(episode_num, trades_list, output_dir="."):
    if not trades_list:
        print(f"No trades in episode {episode_num} to summarize.")
        return
        
    pnls = [t['pnl'] for t in trades_list]
    durations = [t['duration'] for t in trades_list]
    
    n = len(pnls)
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls) if n > 1 else 0
    ci_pnl = 1.96 * (std_pnl / np.sqrt(n)) if n > 0 else 0
    
    # Bayesian Win Rate Calculation
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss > 0:
        bayesian_wr = (gross_profit / gross_loss) - 1
    else:
        bayesian_wr = float('inf')
    
    mean_dur = np.mean(durations)
    std_dur = np.std(durations) if n > 1 else 0
    ci_dur = 1.96 * (std_dur / np.sqrt(n)) if n > 0 else 0

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))
    
    # PnL Histogram
    ax1.hist(pnls, bins=50, color='skyblue', edgecolor='black')
    ax1.axvline(mean_pnl, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_pnl:.2f}')
    ax1.axvline(mean_pnl - ci_pnl, color='orange', linestyle='dotted', linewidth=2, label=f'95% CI: ±{ci_pnl:.2f}')
    ax1.axvline(mean_pnl + ci_pnl, color='orange', linestyle='dotted', linewidth=2)
    ax1.axvline(10.0, color='green', linestyle='-', linewidth=2, label='Target: $10.00 (Raw)')
    ax1.axvline(0.0, color='black', linestyle='-', linewidth=1)
    ax1.set_title(f'Trade Net PnL (Bayesian WR: {bayesian_wr:.2f})')
    ax1.set_xlabel('Net PnL ($)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Duration Histogram
    ax2.hist(durations, bins=50, color='lightgreen', edgecolor='black')
    ax2.axvline(mean_dur, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_dur:.2f} bars')
    ax2.axvline(mean_dur - ci_dur, color='orange', linestyle='dotted', linewidth=2, label=f'95% CI: ±{ci_dur:.2f}')
    ax2.axvline(mean_dur + ci_dur, color='orange', linestyle='dotted', linewidth=2)
    ax2.set_title(f'Trade Duration Distribution (Epoch {episode_num})')
    ax2.set_xlabel('Duration (1m bars)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # Time of Day Scatter Plot
    central_tz = pytz.timezone('US/Central')
    if all('entry_ts' in t for t in trades_list):
        tods = []
        for t in trades_list:
            dt = datetime.datetime.fromtimestamp(t['entry_ts'], tz=datetime.timezone.utc)
            ct = dt.astimezone(central_tz)
            tods.append(ct.hour + ct.minute / 60.0 + ct.second / 3600.0)
            
        colors = ['green' if p >= 0 else 'red' for p in pnls]
        ax3.scatter(tods, pnls, c=colors, alpha=0.6, edgecolors='none')
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.axvline(17, color='blue', linestyle=':', label='Reopen (17:00 CT)')
        ax3.axvline(16, color='blue', linestyle=':', label='Halt (16:00 CT)')
        ax3.set_title(f'PnL vs Time of Day Entry (Epoch {episode_num})')
        ax3.set_xlabel('Time of Day (Hours CT)')
        ax3.set_ylabel('Net PnL ($)')
        ax3.set_xlim(0, 24)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "Entry Time data not available", ha='center', va='center')
        ax3.set_title(f'PnL vs Time of Day Entry (Epoch {episode_num})')
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'epoch_{episode_num}_summary.png')
    plt.savefig(output_path)
    plt.close()
    
    # Save CSV
    csv_path = os.path.join(output_dir, f'epoch_{episode_num}_trades.csv')
    try:
        with open(csv_path, 'w', newline='') as f:
            if trades_list and len(trades_list) > 0:
                writer = csv.DictWriter(f, fieldnames=trades_list[0].keys())
                writer.writeheader()
                writer.writerows(trades_list)
        print(f"Saved trades to {csv_path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")
    
    print(f"=== Epoch {episode_num} Summary ===")
    print(f"Total Trades: {n}")
    print(f"Average Net PnL: ${mean_pnl:.2f} ± ${ci_pnl:.2f}")
    print(f"Average Duration: {mean_dur:.2f} bars ± {ci_dur:.2f} bars")
    print(f"Saved histogram to {output_path}")

def plot_learning_curve(epoch_rewards, epoch_mean_pnls, epoch_entropies=None, output_dir="."):
    if not epoch_rewards:
        return
        
    epochs = range(len(epoch_rewards))
    num_plots = 3 if epoch_entropies else 2
    
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2] if num_plots == 3 else None
    
    # Reward Curve
    ax1.plot(epochs, epoch_rewards, marker='o', color='purple', linestyle='-', linewidth=2)
    ax1.set_title('Total Episode Reward per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Reward ($)')
    ax1.grid(True, alpha=0.3)
    
    # Mean PnL Curve
    if len(epoch_mean_pnls) > 0:
        ax2.plot(epochs, epoch_mean_pnls, marker='o', color='blue', linestyle='-', linewidth=2)
        ax2.axhline(10.0, color='green', linestyle='--', linewidth=2, label='Target: $10.00')
        ax2.axhline(0.0, color='red', linestyle='--', linewidth=1)
        ax2.set_title('Average Net PnL per Trade per Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Net PnL ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
    # ML Quality Metric: Mean Entropy
    if ax3 and len(epoch_entropies) > 0:
        ax3.plot(epochs, epoch_entropies, marker='o', color='darkorange', linestyle='-', linewidth=2)
        ax3.set_title('Mean Policy Entropy per Epoch (ML Quality)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Mean Entropy')
        ax3.grid(True, alpha=0.3)
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mamba_learning_curve.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Updated learning curve at {output_path}")
