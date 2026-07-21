import os
import glob
import json
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8080
INDEX_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')

class TelemetryState:
    def __init__(self):
        self.metrics = {}  # metric_id -> start_value, start_time

    def poll(self):
        files = glob.glob("artifacts/telemetry_*.json")
        active_ids = set()
        current_time = time.time()
        
        results = []

        for f in files:
            try:
                with open(f, "r") as fp:
                    data = json.load(fp)
                
                metric_id = data.get("metric_id")
                if not metric_id:
                    continue
                    
                # Check heartbeat: if no update in 10 seconds, worker is dead/finished
                ts = data.get("timestamp", current_time)
                if current_time - ts > 10.0:
                    try:
                        os.remove(f)
                    except:
                        pass
                    continue
                    
                active_ids.add(metric_id)
                
                current = data.get("current", 0)
                total = data.get("total", 1)
                label = data.get("label", metric_id)
                
                if metric_id not in self.metrics:
                    self.metrics[metric_id] = {'start_value': current, 'start_time': current_time}
                
                start_val = self.metrics[metric_id]['start_value']
                start_time = self.metrics[metric_id]['start_time']
                
                pct = (current / max(1, total)) * 100
                
                elapsed = current_time - start_time
                processed = current - start_val
                
                eta_str = "Calculating..."
                vel_str = "..."
                
                if processed > 0 and elapsed > 0:
                    velocity = processed / elapsed
                    remaining = total - current
                    eta_seconds = remaining / velocity
                    
                    if eta_seconds > 3600:
                        eta_str = f"{eta_seconds/3600:.1f} hours"
                    elif eta_seconds > 60:
                        eta_str = f"{eta_seconds/60:.1f} mins"
                    else:
                        eta_str = f"{eta_seconds:.0f} secs"
                        
                    vel_str = f"{velocity:.1f}/s"
                
                results.append({
                    "metric_id": metric_id,
                    "label": label,
                    "current": current,
                    "total": total,
                    "pct": pct,
                    "eta_str": eta_str,
                    "vel_str": vel_str
                })
                
            except Exception:
                pass # Ignore temporary read locks
                
        # Cleanup deleted
        for metric_id in list(self.metrics.keys()):
            if metric_id not in active_ids:
                del self.metrics[metric_id]
                
        # Sort so they appear consistently
        results.sort(key=lambda x: x['metric_id'])
        return results


telemetry_state = TelemetryState()

class TelemetryHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            with open(INDEX_HTML_PATH, 'rb') as f:
                self.wfile.write(f.read())
                
        elif self.path == '/api/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            metrics = telemetry_state.poll()
            self.wfile.write(json.dumps(metrics).encode('utf-8'))
            
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress standard logging to keep terminal clean
        pass


def run():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, TelemetryHandler)
    print(f"[Telemetry] Server starting on http://localhost:{PORT}")
    
    # Open the user's default browser
    try:
        webbrowser.open(f"http://localhost:{PORT}")
    except:
        pass
        
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[Telemetry] Server stopping...")
    finally:
        httpd.server_close()

if __name__ == '__main__':
    run()
