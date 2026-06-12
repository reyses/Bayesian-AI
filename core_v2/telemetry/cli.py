import os
import glob
import json
import time
import sys

def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        # Fallback to ANSI if TERM is not set
        if 'TERM' in os.environ:
            os.system('clear')
        else:
            print('\033c', end='')

def main():
    print("Watching telemetry files in artifacts/...")
    last_update_ts = {}
    start_values = {}
    start_times = {}
    
    try:
        while True:
            files = glob.glob("artifacts/telemetry_*.json")
            if not files:
                clear_screen()
                print("No active telemetry streams found. Waiting...")
                time.sleep(3.0)
                continue
                
            lines_to_print = []
            lines_to_print.append("=== Bayesian-AI Live CLI Telemetry ===")
            lines_to_print.append(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            current_time = time.time()
            active_any = False
            
            for f in sorted(files):
                try:
                    with open(f, "r") as fp:
                        data = json.load(fp)
                    metric_id = data.get("metric_id")
                    if not metric_id:
                        continue
                        
                    ts = data.get("timestamp", current_time)
                    # heartbeat check (15 seconds)
                    if current_time - ts > 15.0:
                        try:
                            os.remove(f)
                        except:
                            pass
                        continue
                        
                    current = data.get("current", 0.0)
                    total = data.get("total", 1.0)
                    label = data.get("label", metric_id)
                    
                    if metric_id not in start_values:
                        start_values[metric_id] = current
                        start_times[metric_id] = current_time
                        
                    pct = (current / max(1.0, total)) * 100.0
                    bar_width = 30
                    filled_len = int(bar_width * current // max(1.0, total))
                    bar = '█' * filled_len + '░' * (bar_width - filled_len)
                    
                    elapsed = current_time - start_times[metric_id]
                    processed = current - start_values[metric_id]
                    
                    if processed > 0 and elapsed > 0:
                        velocity = processed / elapsed
                        remaining = total - current
                        eta_seconds = remaining / velocity
                        
                        if eta_seconds > 3600:
                            eta_str = f"{eta_seconds/3600:.1f}h"
                        elif eta_seconds > 60:
                            eta_str = f"{eta_seconds/60:.1f}m"
                        else:
                            eta_str = f"{eta_seconds:.0f}s"
                        vel_str = f"{velocity:.1f}/s"
                    else:
                        eta_str = "Calculating..."
                        vel_str = "..."
                        
                    lines_to_print.append(f"📌 {label} ({metric_id})")
                    lines_to_print.append(f"   [{bar}] {pct:.1f}% ({int(current)}/{int(total)})")
                    lines_to_print.append(f"   Speed: {vel_str} | ETA: {eta_str} | Elapsed: {elapsed:.0f}s\n")
                    active_any = True
                except Exception as e:
                    pass
            
            if active_any:
                clear_screen()
                print("\n".join(lines_to_print))
            
            # Slowed down from 1.0 to 3.0 to stop updating the terminal so fast!
            time.sleep(3.0)
    except KeyboardInterrupt:
        print("\nExiting telemetry viewer.")

if __name__ == "__main__":
    main()
