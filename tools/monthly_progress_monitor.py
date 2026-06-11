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
        print("[WARNING] Telegram credentials not found.")
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Monitor monthly progression of segmentation and alert on completion.")
    parser.add_argument('--repo-root', type=str, default='/home/reyse/Bayesian-AI', help="Path to repository root on VM")
    args = parser.parse_args()

    repo_root = args.repo_root
    artifacts_dir = os.path.join(repo_root, 'artifacts')
    l0_dir = os.path.join(repo_root, 'DATA', 'ATLAS', 'FEATURES_5s_v2', 'L0')
    
    if not os.path.exists(l0_dir):
        # Fallback to local path if running locally
        l0_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DATA', 'ATLAS', 'FEATURES_5s_v2', 'L0'))
        artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
        
    if not os.path.exists(l0_dir):
        print(f"Error: Raw L0 directory '{l0_dir}' does not exist.")
        sys.exit(1)

    # 1. Get all expected days from parquet files
    parquet_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    expected_days = [os.path.basename(f).replace('.parquet', '') for f in parquet_files]
    
    # 2. Group expected days by month (YYYY_MM)
    months_map = {}
    for day in expected_days:
        m = day[:7]  # YYYY_MM
        if m not in months_map:
            months_map[m] = []
        months_map[m].append(day)
        
    # 3. Check completed days in artifacts
    completed_days = set()
    stage1_files = glob.glob(os.path.join(artifacts_dir, 'stage1_segments_*.json'))
    for f in stage1_files:
        day = os.path.basename(f).replace('stage1_segments_', '').replace('.json', '')
        completed_days.add(day)

    # 4. Load already alerted months
    state_file = os.path.join(artifacts_dir, 'alerted_months.json')
    alerted_months = set()
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                alerted_months = set(json.load(f))
        except: pass

    # 5. Check completion for each month
    print("=== Monthly Progression Monitor ===")
    newly_completed = []
    
    for m in sorted(months_map.keys()):
        all_days = months_map[m]
        finished_days = [d for d in all_days if d in completed_days]
        pct = (len(finished_days) / len(all_days)) * 100.0
        
        print(f"Month {m}: {len(finished_days)}/{len(all_days)} completed ({pct:.1f}%)")
        
        if len(finished_days) == len(all_days):
            if m not in alerted_months:
                newly_completed.append(m)
                alerted_months.add(m)

    # 6. Save alerted state and send telegram alerts
    if newly_completed:
        # Save updated state
        try:
            with open(state_file, 'w') as f:
                json.dump(list(alerted_months), f)
        except Exception as e:
            print("Failed to save state:", e)
            
        for m in newly_completed:
            msg = f"📬 *Month {m} Stage 1 scan completed!* (All {len(months_map[m])} days scanned successfully.)"
            print(f"Sending Telegram alert for newly completed month: {m}")
            send_telegram_alert(msg)

if __name__ == '__main__':
    main()
