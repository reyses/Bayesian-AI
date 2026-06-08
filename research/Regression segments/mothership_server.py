import os
import glob
import time
import zipfile
import io
from flask import Flask, jsonify, request, send_file
import werkzeug

app = Flask(__name__)

ATLAS_ROOT = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
ARTIFACTS_DIR = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/artifacts"

# In-memory lock dictionary: { "2025_02_05": timestamp }
locked_jobs = {}
LOCK_TIMEOUT = 3600 * 6  # 6 hours (if a drone dies, job unlocks after 6h)

@app.route('/')
def index():
    return "Mothership Server is RUNNING."

@app.route('/get_job', methods=['GET'])
def get_job():
    l0_dir = os.path.join(ATLAS_ROOT, 'FEATURES_5s_v2', 'L0')
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    all_days = [os.path.basename(f).replace('FEATURES_', '').replace('.parquet', '') for f in files]
    
    # Check completed
    completed_files = glob.glob(os.path.join(ARTIFACTS_DIR, 'stage2_segments_*.json'))
    completed_days = [os.path.basename(f).replace('stage2_segments_', '').replace('.json', '') for f in completed_files]
    
    now = time.time()
    
    for day in all_days:
        if day in completed_days:
            continue
            
        if day in locked_jobs:
            if now - locked_jobs[day] < LOCK_TIMEOUT:
                continue # Still locked by a drone
        
        # We found a valid day!
        locked_jobs[day] = now
        print(f"[MOTHERSHIP] Assigned Job {day} to Drone.")
        return jsonify({"day": day})
        
    return jsonify({"day": None, "message": "All jobs completed or locked!"})

@app.route('/download/<day>', methods=['GET'])
def download_data(day):
    """Dynamically packages the 5s OHLCV and all feature layers for the requested day into a ZIP stream."""
    memory_file = io.BytesIO()
    
    paths_to_zip = {
        f"DATA/ATLAS/5s/{day}.parquet": os.path.join(ATLAS_ROOT, '5s', f'{day}.parquet'),
        f"DATA/ATLAS/FEATURES_5s_v2/L0/FEATURES_{day}.parquet": os.path.join(ATLAS_ROOT, 'FEATURES_5s_v2', 'L0', f'FEATURES_{day}.parquet'),
        f"DATA/ATLAS/FEATURES_5s_v2/L1/FEATURES_{day}.parquet": os.path.join(ATLAS_ROOT, 'FEATURES_5s_v2', 'L1', f'FEATURES_{day}.parquet'),
        f"DATA/ATLAS/FEATURES_5s_v2/L2/FEATURES_{day}.parquet": os.path.join(ATLAS_ROOT, 'FEATURES_5s_v2', 'L2', f'FEATURES_{day}.parquet'),
        f"DATA/ATLAS/FEATURES_5s_v2/L3/FEATURES_{day}.parquet": os.path.join(ATLAS_ROOT, 'FEATURES_5s_v2', 'L3', f'FEATURES_{day}.parquet'),
    }
    
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for arcname, abs_path in paths_to_zip.items():
            if os.path.exists(abs_path):
                zf.write(abs_path, arcname)
            else:
                print(f"[MOTHERSHIP] Warning: Missing file {abs_path}")
                
    memory_file.seek(0)
    print(f"[MOTHERSHIP] Streaming packaged dataset for {day} to Drone.")
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'drone_data_{day}.zip'
    )

@app.route('/submit/<day>', methods=['POST'])
def submit_job(day):
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and file.filename.endswith('.json'):
        save_path = os.path.join(ARTIFACTS_DIR, f"stage2_segments_{day}.json")
        file.save(save_path)
        print(f"[MOTHERSHIP] SUCCESS! Received finished job for {day} from Drone.")
        
        if day in locked_jobs:
            del locked_jobs[day]
            
        return jsonify({"status": "success", "message": f"Saved {day}"})
        
    return jsonify({"error": "Invalid file"}), 400

if __name__ == '__main__':
    print("\n=============================================")
    print("   MOTHERSHIP SERVER INITIALIZED")
    print("   Awaiting Drone Connections...")
    print("=============================================\n")
    # Bind to 0.0.0.0 to allow LAN connections
    app.run(host='0.0.0.0', port=5050, debug=False)
