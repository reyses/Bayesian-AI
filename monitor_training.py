import os
import time
import json
import requests
import re
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

LOG_PATH = r"C:\Users\reyse\.gemini\antigravity\brain\1be668b8-a86a-4545-9048-dfbd1982889c\.system_generated\tasks\task-16357.log"
OOS_DATA_PATH = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\training\rl_engine\oos_trade_data.json"
ENV_PATH = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\.env"

def load_env():
    env_vars = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    env_vars[k.strip()] = v.strip()
    return env_vars

def get_credentials():
    env = load_env()
    token = env.get("TELEGRAM_BOT_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN"))
    chat_id = env.get("TELEGRAM_CHAT_ID", os.environ.get("TELEGRAM_CHAT_ID"))
    return token, chat_id

def send_telegram(message):
    token, chat_id = get_credentials()
    if not token or not chat_id:
        print("Missing telegram credentials in .env")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload)
        print(f"[{time.strftime('%X')}] Sent text. Status: {res.status_code}")
    except Exception as e:
        print(f"Telegram error: {e}")

def send_telegram_photo(buf, caption=""):
    token, chat_id = get_credentials()
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    buf.seek(0)
    try:
        res = requests.post(url, data={"chat_id": chat_id, "caption": caption},
                            files={"photo": ("chart.png", buf, "image/png")})
        print(f"[{time.strftime('%X')}] Sent photo. Status: {res.status_code}")
    except Exception as e:
        print(f"Telegram photo error: {e}")

def build_distribution_chart(oos_data):
    """Builds a 1x2 matplotlib figure: Trade PnL and Trade Duration distributions with fitted normal."""
    pnls = np.array(oos_data['pnls'])
    durs = np.array(oos_data['durations'])
    segment = oos_data.get('segment', '?')
    eval_dates = oos_data.get('eval_dates', '')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')
    
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#e0e0e0')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # --- Panel 1: Trade PnL Distribution ---
    ax1 = axes[0]
    clipped = np.clip(pnls, np.percentile(pnls, 1), np.percentile(pnls, 99))
    n_bins = min(80, max(30, len(clipped) // 50))
    counts, bin_edges, patches = ax1.hist(clipped, bins=n_bins, density=True,
                                           color='#4a9eed', alpha=0.6, edgecolor='none')
    # Color wins/losses
    for patch, left in zip(patches, bin_edges[:-1]):
        patch.set_facecolor('#e74c3c' if left < 0 else '#2ecc71')
        patch.set_alpha(0.65)

    # Fit normal
    mu, sigma = stats.norm.fit(pnls)
    x = np.linspace(clipped.min(), clipped.max(), 400)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), color='#f39c12', lw=2.5,
             label=f'Normal fit\nμ={mu:.1f}, σ={sigma:.1f}')
    ax1.axvline(0, color='white', lw=1, ls='--', alpha=0.5)
    ax1.axvline(mu, color='#f39c12', lw=1, ls=':', alpha=0.7)
    
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    pf_str = f"{np.sum(wins)/max(np.sum(np.abs(losses)),1e-9):.4f}"
    
    ax1.set_title(f'OOS Trade PnL — Seg {segment}\n{eval_dates}', color='white', fontsize=12, pad=10)
    ax1.set_xlabel('PnL ($)', color='#aaa')
    ax1.set_ylabel('Density', color='#aaa')
    ax1.legend(facecolor='#0f3460', labelcolor='white', fontsize=9)
    ax1.text(0.98, 0.97, f'n={len(pnls):,}  PF={pf_str}', transform=ax1.transAxes,
             ha='right', va='top', color='#ccc', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f3460', alpha=0.7))

    # --- Panel 2: Trade Duration Distribution ---
    ax2 = axes[1]
    dur_clipped = np.clip(durs, 0, np.percentile(durs, 99))
    n_bins2 = min(60, max(20, int(dur_clipped.max() - dur_clipped.min()) // 2 + 1))
    ax2.hist(dur_clipped, bins=n_bins2, density=True,
             color='#9b59b6', alpha=0.65, edgecolor='none')
    
    # Fit normal to duration
    mu_d, sigma_d = stats.norm.fit(durs)
    x2 = np.linspace(max(0, dur_clipped.min()), dur_clipped.max(), 400)
    ax2.plot(x2, stats.norm.pdf(x2, mu_d, sigma_d), color='#1abc9c', lw=2.5,
             label=f'Normal fit\nμ={mu_d:.1f}s, σ={sigma_d:.1f}s')
    ax2.axvline(mu_d, color='#1abc9c', lw=1, ls=':', alpha=0.7)
    
    ax2.set_title(f'OOS Trade Duration — Seg {segment}\n{eval_dates}', color='white', fontsize=12, pad=10)
    ax2.set_xlabel('Duration (bars)', color='#aaa')
    ax2.set_ylabel('Density', color='#aaa')
    ax2.legend(facecolor='#0f3460', labelcolor='white', fontsize=9)
    ax2.text(0.98, 0.97, f'median={np.median(durs):.0f}  mean={mu_d:.0f}',
             transform=ax2.transAxes, ha='right', va='top', color='#ccc', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f3460', alpha=0.7))

    plt.tight_layout(pad=2.0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

def parse_log_stats():
    """Parse the training log for current status and pooled OOS aggregate."""
    if not os.path.exists(LOG_PATH):
        return None
    try:
        with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return None
    if not lines:
        return None

    current_segment = "?"
    for line in reversed(lines):
        if "WALK-FORWARD STEP" in line:
            m = re.search(r"STEP (\d+)", line)
            if m:
                current_segment = m.group(1)
            break

    active_status = "Unknown"
    for line in reversed(lines):
        if "[EVAL] Running OOS Evaluation" in line:
            active_status = "⏳ Evaluating OOS"
            break
        if "[TRAIN] Running Epoch" in line:
            m = re.search(r"Epoch (\d+/\d+)", line)
            if m:
                active_status = f"🏋️ Training Epoch {m.group(1)}"
            break

    # Last IS loss
    last_loss = None
    for line in reversed(lines):
        if "[DIAGNOSTICS] Avg Loss:" in line:
            m = re.search(r"Avg Loss: ([\d.]+)", line)
            if m:
                last_loss = float(m.group(1))
            break

    # Last completed OOS diag lines
    oos_seg_diag = []
    for idx in range(len(lines)-1, -1, -1):
        if "[OOS DIAGNOSTICS]" in lines[idx]:
            # We found the last one. Go backwards to find the start of the block.
            start_idx = idx
            while start_idx > 0 and "[OOS DIAGNOSTICS]" in lines[start_idx-1]:
                start_idx -= 1
            # Collect all lines in the block
            for j in range(start_idx, idx+1):
                oos_seg_diag.append(lines[j].strip().replace("[OOS DIAGNOSTICS] ", ""))
            break
    oos_seg_diag = "\n".join(oos_seg_diag)

    # Pooled aggregate
    pooled = {}
    for idx in range(len(lines)-1, -1, -1):
        if "[POOLED AGGREGATE]" in lines[idx]:
            for j in range(idx+1, min(idx+12, len(lines))):
                l = lines[j].strip()
                if not l or "===" in l:
                    break
                parts = l.split(":", 1)
                if len(parts) == 2:
                    pooled[parts[0].strip()] = parts[1].strip()
            break

    return {
        'segment': current_segment,
        'status': active_status,
        'last_loss': last_loss,
        'oos_seg_diag': oos_seg_diag,
        'pooled': pooled,
    }

def format_message(stats, oos_data=None):
    pooled = stats.get('pooled', {})
    pf = pooled.get('profit_factor', 'N/A')
    net_pnl = pooled.get('total_net_pnl', 'N/A')
    trades = pooled.get('total_trades', 'N/A')
    maxdd = pooled.get('pooled_max_drawdown', 'N/A')
    mean_pnl = pooled.get('pooled_mean_pnl', 'N/A')
    mean_ci = pooled.get('pooled_mean_pnl_ci', 'N/A')
    cap_avail = pooled.get('pooled_cap_vs_avail', 'N/A')

    try:
        pf_val = float(pf)
        pf_emoji = "🟢" if pf_val >= 1.0 else "🔴"
        pf_str = f"{pf_emoji} {pf_val:.4f}"
    except:
        pf_str = str(pf)

    try:
        net = float(net_pnl)
        net_str = f"{'🟢' if net >= 0 else '🔴'} ${net:,.2f}"
    except:
        net_str = str(net_pnl)

    msg = f"📊 *Curriculum Monitor — Auto Update*\n\n"
    msg += f"*Segment:* {stats['segment']}  |  *Status:* {stats['status']}\n"
    if stats['last_loss'] is not None:
        msg += f"*Last IS Loss:* `{stats['last_loss']:.4f}`\n"
    msg += f"\n─── Last OOS Segment ───\n"
    if stats['oos_seg_diag']:
        msg += f"`{stats['oos_seg_diag']}`\n"
    msg += f"\n─── Pooled Grand Aggregate ───\n"
    msg += f"*Profit Factor:* {pf_str}\n"
    msg += f"*Net PnL:* {net_str}\n"
    msg += f"*Total Trades:* {trades}\n"
    msg += f"*Max Drawdown:* ${float(maxdd):,.2f}\n" if maxdd != 'N/A' else f"*Max Drawdown:* N/A\n"
    msg += f"*Mean PnL/trade:* `{mean_pnl}` CI: `{mean_ci}`\n"
    msg += f"*Cap vs Avail:* `{cap_avail}`\n"
    if oos_data:
        msg += f"\n📈 Chart: Seg {oos_data['segment']} OOS distributions attached"
    return msg

def run_update():
    stats = parse_log_stats()
    if not stats:
        send_telegram("⚠️ Monitor: Could not read training log.")
        return

    oos_data = None
    if os.path.exists(OOS_DATA_PATH):
        try:
            with open(OOS_DATA_PATH, 'r') as f:
                oos_data = json.load(f)
        except Exception:
            oos_data = None

    msg = format_message(stats, oos_data)
    send_telegram(msg)

    if oos_data and len(oos_data.get('pnls', [])) > 10:
        try:
            buf = build_distribution_chart(oos_data)
            caption = f"OOS Seg {oos_data['segment']} | {oos_data.get('eval_dates','')}"
            send_telegram_photo(buf, caption)
        except Exception as e:
            print(f"Chart error: {e}")
            send_telegram(f"⚠️ Chart generation failed: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--run-once":
        print("Running one-time background update to Telegram...")
        run_update()
    else:
        print("Starting background training monitor (runs every 20 mins)...")
        run_update()  # Immediate on boot
        while True:
            time.sleep(1200)
            run_update()
