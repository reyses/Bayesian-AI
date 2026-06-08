import sys
import os
import json
import numpy as np
import scipy.stats as stats
from pathlib import Path

# Setup Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
from telegram_mcp import send_telegram_alert

def send_autostats(segment_arg):
    trade_data_path = SCRIPT_DIR / "oos_trade_data.json"
    if not trade_data_path.exists():
        send_telegram_alert(f"Autostats Error: No OOS trade data found at {trade_data_path}")
        return
        
    with open(trade_data_path, "r") as f:
        data = json.load(f)
        
    pnls = np.array(data.get('pnls', []))
    durations = np.array(data.get('durations', []))
    
    if len(pnls) == 0:
        send_telegram_alert(f"Autostats Error: No PnLs to calculate for segment {segment_arg}.")
        return
        
    mu, std = stats.norm.fit(pnls)
    skew = stats.skew(pnls)
    kurt = stats.kurtosis(pnls)
    gross_profits = np.sum(pnls[pnls > 0])
    gross_losses = np.abs(np.sum(pnls[pnls < 0]))
    pf = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    net_pnl = np.sum(pnls)
    
    win_rate = (len(pnls[pnls > 0]) / len(pnls)) * 100
    
    stats_msg = (
        f"📊 **Autostats for Segment {segment_arg}** 📊\n\n"
        f"**Dates:** {data.get('eval_dates', 'Unknown')}\n"
        f"**Trades:** {len(pnls)}\n"
        f"**Net PnL:** ${net_pnl:.2f}\n"
        f"**Win Rate:** {win_rate:.1f}%\n"
        f"**Profit Factor:** {pf:.3f}\n\n"
        f"**Distribution:**\n"
        f"μ={mu:.3f} | σ={std:.2f}\n"
        f"Skew={skew:+.2f} | Kurt={kurt:+.2f}\n"
    )
    
    res = send_telegram_alert(stats_msg)
    print(f"[Telegram] {res}")

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "latest"
    send_autostats(arg)
    # TODO: Implement the stats logic here
