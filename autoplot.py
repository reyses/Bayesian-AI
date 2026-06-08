import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

# Setup Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
from telegram_mcp import send_telegram_media

def send_autoplot(segment_arg):
    trade_data_path = SCRIPT_DIR / "oos_trade_data.json"
    if not trade_data_path.exists():
        print("No OOS trade data found.")
        return None
        
    with open(trade_data_path, "r") as f:
        data = json.load(f)
        
    pnls = np.array(data.get('pnls', []))
    durations = np.array(data.get('durations', []))
    
    if len(pnls) == 0:
        print("No PnLs to plot.")
        return None
        
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#1a1c23')
    ax1.set_facecolor('#1a1c23')
    ax2.set_facecolor('#1a1c23')
    
    fig.suptitle(f"OOS Trade Diagnostics — Seg {data.get('segment', segment_arg)} · eval: {data.get('eval_dates', 'Unknown')}", fontsize=14, color='gold')
    
    # --- PnL Histogram ---
    ax1.set_title(f"PnL (real) - Distribution (n={len(pnls)})", fontsize=12, color='lightgray')
    
    q_low = np.percentile(pnls, 1) if len(pnls) > 0 else 0
    q_high = np.percentile(pnls, 99) if len(pnls) > 0 else 0
    pnl_filtered = pnls[(pnls >= q_low) & (pnls <= q_high)]
    if len(pnl_filtered) == 0:
        pnl_filtered = pnls
        
    N, bins, patches = ax1.hist(pnl_filtered, bins=50, density=True, alpha=0.9)
    for bin_val, patch in zip(bins, patches):
        if bin_val < 0:
            patch.set_facecolor('#c24a4a') # indianred
        else:
            patch.set_facecolor('#39a66d') # mediumseagreen
            
    # Overlay Normal Dist
    if len(pnl_filtered) > 1:
        mu, std = stats.norm.fit(pnl_filtered)
        xmin, xmax = ax1.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax1.plot(x, p, 'w-', linewidth=1.5, alpha=0.6)
        
        ax1.axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
        ax1.axvline(mu, color='cyan', linestyle=':', linewidth=1.5, alpha=0.8)
        
        skew = stats.skew(pnls)
        kurt = stats.kurtosis(pnls)
        gross_profits = np.sum(pnls[pnls > 0])
        gross_losses = np.abs(np.sum(pnls[pnls < 0]))
        pf = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        net_pnl = np.sum(pnls)
        
        stats_text = (f"μ={mu:.3f}  σ={std:.2f}  skew={skew:+.2f}\n"
                      f"excess kurt={kurt:+.2f}  PF={pf:.4f}\n"
                      f"Net=${net_pnl:.0f}")
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10, color='lightgray',
                verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle='round', facecolor='#12233f', edgecolor='#436087', alpha=0.8))
        
    ax1.set_xlabel("PnL ($)", color='lightgray')
    ax1.set_ylabel("Density", color='lightgray')
    ax1.tick_params(colors='lightgray')
    
    # --- Duration Histogram ---
    ax2.set_title("Duration (real, bars)", fontsize=12, color='lightgray')
    if len(durations) > 0:
        dur_filtered = durations[durations <= np.percentile(durations, 99)]
        ax2.hist(dur_filtered, bins=30, density=True, color='#7c5295', alpha=0.9)
        
        median_dur = np.median(durations)
        mean_dur = np.mean(durations)
        
        ax2.axvline(median_dur, color='cyan', linestyle='-', linewidth=1.5, alpha=0.8, label=f'median={median_dur:.1f}')
        ax2.axvline(mean_dur, color='gold', linestyle=':', linewidth=1.5, alpha=0.8, label=f'mean={mean_dur:.1f}')
        legend = ax2.legend(facecolor='#12233f', edgecolor='#436087')
        for text in legend.get_texts():
            text.set_color('lightgray')
            
    ax2.set_xlabel("Duration (bars)", color='lightgray')
    ax2.set_ylabel("Density", color='lightgray')
    ax2.tick_params(colors='lightgray')
    
    chart_path = SCRIPT_DIR / "autoplot_chart.png"
    plt.tight_layout()
    plt.savefig(chart_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    
    res = send_telegram_media(str(chart_path), caption=f"📊 Autoplot for Segment {segment_arg}")
    print(f"[Telegram] {res}")

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "latest"
    send_autoplot(arg)
    # TODO: Implement the plotting logic here
