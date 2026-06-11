import os
import sys
import glob
import json
import time

def load_env():
    # Look for .env in current dir, or parent dirs
    paths = ['.env', '../../.env', '../.env', 'Bayesian-AI/.env']
    for p in paths:
        abs_p = os.path.abspath(os.path.join(os.path.dirname(__file__), p))
        if os.path.exists(abs_p):
            try:
                with open(abs_p, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, val = line.split('=', 1)
                            os.environ[key.strip()] = val.strip()
                break
            except Exception as e:
                print(f"[WARNING] Error reading env file {abs_p}: {e}")

def send_telegram_alert(message):
    load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[WARNING] Telegram credentials (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID) not found in env or .env file.")
        return False
        
    import urllib.request
    import urllib.parse
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    data = urllib.parse.urlencode(payload).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res.get("ok", False)
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram alert: {e}")
        return False

def get_telemetry_data(artifacts_dir):
    fpaths = glob.glob(os.path.join(artifacts_dir, 'telemetry_*.json'))
    d1 = {}
    for f in fpaths:
        try:
            with open(f, 'r') as fp:
                d1[f] = json.load(fp)
        except: pass
    return d1

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Beam active VM telemetry directly to Telegram.")
    parser.add_argument('--dir', type=str, default='artifacts', help="Directory containing telemetry JSON files")
    args = parser.parse_args()

    artifacts_dir = args.dir
    if not os.path.exists(artifacts_dir):
        print(f"Error: Directory '{artifacts_dir}' does not exist.")
        sys.exit(1)

    print("Taking first telemetry snapshot...")
    d1 = get_telemetry_data(artifacts_dir)
    t1 = time.time()
    
    time.sleep(2.0)
    
    print("Taking second telemetry snapshot...")
    d2 = get_telemetry_data(artifacts_dir)
    t2 = time.time()
    dt = t2 - t1

    results = []
    for f in d2:
        if f in d1:
            results.append({'d1': d1[f], 'd2': d2[f], 'dt': dt})

    if not results:
        msg = "No active VM telemetry streams found right now."
        print(msg)
        send_telegram_alert(msg)
        return

    lines = ["📊 *Bayesian-AI Live CLI Telemetry*"]
    lines.append(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    current_time = time.time()
    for item in sorted(results, key=lambda x: x['d2'].get('metric_id', '')):
        data1 = item['d1']
        data2 = item['d2']
        delta_t = item['dt']
        
        ts = data2.get("timestamp", current_time)
        if current_time - ts > 120.0:
            continue  # dead stream

        metric_id = data2.get("metric_id")
        current = data2.get("current", 0.0)
        total = data2.get("total", 1.0)
        label = data2.get("label", metric_id)
        
        pct = (current / max(1.0, total)) * 100.0
        bar_width = 15
        filled_len = int(bar_width * current // max(1.0, total))
        bar = '█' * filled_len + '░' * (bar_width - filled_len)
        
        c1 = data1.get("current", 0.0)
        c2 = data2.get("current", 0.0)
        processed = c2 - c1
        
        if processed >= 0 and delta_t > 0:
            velocity = processed / delta_t
            remaining = total - current
            
            if velocity > 0:
                eta_seconds = remaining / velocity
                if eta_seconds > 3600:
                    eta_str = f"{eta_seconds/3600:.1f}h"
                elif eta_seconds > 60:
                    eta_str = f"{eta_seconds/60:.1f}m"
                else:
                    eta_str = f"{eta_seconds:.0f}s"
                vel_str = f"{velocity:.1f}/s"
            else:
                eta_str = "Static"
                vel_str = "0.0/s"
        else:
            eta_str = "Calculating..."
            vel_str = "..."
            
        lines.append(f"📌 *{label}*")
        lines.append(f"   `[{bar}] {pct:.1f}%` ({int(current)}/{int(total)})")
        lines.append(f"   Speed: {vel_str} | ETA: {eta_str}")
        lines.append("")

    full_text = "\n".join(lines)
    print(full_text)
    
    print("\nSending telemetry snapshot directly to Telegram...")
    ok = send_telegram_alert(full_text)
    if ok:
        print("Telemetry snapshot sent successfully!")
    else:
        print("Telemetry snapshot failed to send.")

if __name__ == '__main__':
    main()
