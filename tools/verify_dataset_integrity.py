import os
import sys
import glob
import json
import traceback

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
    
    # Escape Markdown V1 control characters safely or keep simple format
    # Simple formatting: escape backticks, stars, etc. or send as plain text
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

def verify_json_file(file_path):
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, "File is completely empty (0 bytes)"
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return True, data
    except json.JSONDecodeError as e:
        return False, f"JSON Decode Error: {str(e)}"
    except Exception as e:
        return False, f"Read Error: {str(e)}"

def analyze_day_segments(segments, file_type="stage2"):
    issues = []
    if not isinstance(segments, list):
        return {"error": "Root JSON is not a list", "issues": ["Root JSON is not a list"], "stats": {}}
        
    if len(segments) == 0:
        return {"error": "Empty segment list", "issues": ["Segment list is empty"], "stats": {}}

    stats = {
        "total_segments": len(segments),
        "status_counts": {},
        "total_length": 0,
        "max_residual": 0.0,
    }
    
    try:
        sorted_segs = sorted(segments, key=lambda x: x.get('start_idx', 0))
    except Exception as e:
        return {"error": f"Failed to sort segments: {str(e)}", "issues": [f"Sort error: {str(e)}"], "stats": {}}

    prev_end = None
    
    for idx, seg in enumerate(sorted_segs):
        for key in ['start_idx', 'end_idx', 'length', 'status']:
            if key not in seg:
                issues.append(f"Segment {idx} is missing key: '{key}'")
                
        start = seg.get('start_idx')
        end = seg.get('end_idx')
        length = seg.get('length')
        status = seg.get('status', 'UNKNOWN')
        
        stats["status_counts"][status] = stats["status_counts"].get(status, 0) + 1
        
        if start is not None and end is not None:
            stats["total_length"] += (end - start)
            
            if start > end:
                issues.append(f"Segment {idx} has start_idx ({start}) > end_idx ({end})")
            if length is not None and (end - start) != length:
                issues.append(f"Segment {idx} length mismatch: end_idx - start_idx = {end - start}, but length is {length}")
                
            if prev_end is not None:
                if start < prev_end:
                    issues.append(f"Overlap detected: segment start_idx ({start}) is less than previous end_idx ({prev_end})")
            prev_end = end
            
        max_res = seg.get('max_residual')
        if max_res is not None and max_res != 9999.0:
            stats["max_residual"] = max(stats["max_residual"], max_res)

    return {
        "issues": issues,
        "stats": stats
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify integrity of segmentation outputs on local/VM environment.")
    parser.add_argument('--dir', type=str, default='artifacts', help="Directory containing segment JSON files")
    parser.add_argument('--telegram', action='store_true', help="Send final summary report to Telegram directly")
    args = parser.parse_args()

    artifacts_dir = args.dir
    if not os.path.exists(artifacts_dir):
        print(f"Error: Directory '{artifacts_dir}' does not exist.")
        sys.exit(1)

    # Accumulate log output to send to Telegram if requested
    report_lines = []
    def log(msg):
        print(msg)
        report_lines.append(msg)

    log(f"==================================================")
    log(f"[*] Dataset Integrity Auditor")
    log(f"Scanning: {artifacts_dir}")
    log(f"==================================================")

    stage1_files = sorted(glob.glob(os.path.join(artifacts_dir, 'stage1_segments_*.json')))
    stage2_files = sorted(glob.glob(os.path.join(artifacts_dir, 'stage2_segments_*.json')))

    log(f"Found {len(stage1_files)} Stage 1 segment files.")
    log(f"Found {len(stage2_files)} Stage 2 segment files.")
    log(f"--------------------------------------------------")

    corrupt_files = []
    empty_files = []
    problematic_files = []
    
    # Track global stats
    stage2_total_days = 0
    stage2_total_segments = 0
    stage2_status_totals = {}
    stage2_total_bars = 0

    log("\n[Auditing Stage 2 Files]")
    for fpath in stage2_files:
        fname = os.path.basename(fpath)
        success, res = verify_json_file(fpath)
        if not success:
            log(f"[ERROR] {fname}: CORRUPTED - {res}")
            corrupt_files.append((fname, res))
            continue
            
        analysis = analyze_day_segments(res, "stage2")
        if "error" in analysis:
            if "Empty" in analysis["error"]:
                log(f"[WARNING] {fname}: EMPTY ARRAY")
                empty_files.append(fname)
            else:
                log(f"[ERROR] {fname}: ERROR - {analysis['error']}")
                corrupt_files.append((fname, analysis["error"]))
            continue
            
        issues = analysis["issues"]
        stats = analysis["stats"]
        
        stage2_total_days += 1
        stage2_total_segments += stats["total_segments"]
        stage2_total_bars += stats["total_length"]
        for stat, val in stats["status_counts"].items():
            stage2_status_totals[stat] = stage2_status_totals.get(stat, 0) + val

        if issues:
            log(f"[WARNING] {fname}: Passed JSON parsing but has {len(issues)} logic issues")
            for iss in issues[:3]:
                log(f"   - {iss}")
            if len(issues) > 3:
                log(f"   - ... and {len(issues) - 3} more issues")
            problematic_files.append((fname, issues))

    # Process Stage 1 Files
    stage1_corrupt = 0
    stage1_empty = 0
    for fpath in stage1_files:
        fname = os.path.basename(fpath)
        success, res = verify_json_file(fpath)
        if not success:
            corrupt_files.append((fname, f"Stage 1: {res}"))
            stage1_corrupt += 1
        elif not res:
            empty_files.append(fname)
            stage1_empty += 1

    log(f"\n==================================================")
    log(f"[STATS] SUMMARY REPORT")
    log(f"==================================================")
    log(f"Stage 1 Checked:      {len(stage1_files)}")
    log(f"  - Corrupt:          {stage1_corrupt}")
    log(f"  - Empty:            {stage1_empty}")
    log(f"Stage 2 Checked:      {len(stage2_files)}")
    log(f"  - Clean & Valid:    {stage2_total_days - len(problematic_files)}")
    log(f"  - Logic Issues:     {len(problematic_files)}")
    log(f"  - Empty Arrays:     {len(empty_files) - stage1_empty}")
    log(f"  - Corrupted JSON:   {len(corrupt_files) - stage1_corrupt}")
    
    if stage2_total_days > 0:
        avg_segs = stage2_total_segments / stage2_total_days
        avg_bars = stage2_total_bars / stage2_total_days
        log(f"\n[Stage 2 Metrics]")
        log(f"Total Segments:       {stage2_total_segments}")
        log(f"Avg Segments / Day:   {avg_segs:.1f}")
        log(f"Avg Covered Bars/Day: {avg_bars:.1f}")
        log(f"Status distribution:")
        for status, val in sorted(stage2_status_totals.items()):
            pct = 100 * val / stage2_total_segments
            log(f"  - {status:<20}: {val:>5} ({pct:.1f}%)")
            
    log(f"--------------------------------------------------")
    if not corrupt_files and not empty_files and not problematic_files:
        log("SUCCESS: INTEGRITY CHECK PASSED (All parsed files are valid and healthy!)")
    else:
        log("WARNING: INTEGRITY CHECK FAILED / WARNINGS FOUND")
        if corrupt_files:
            log(f"  [ERROR] {len(corrupt_files)} CORRUPT FILES FOUND:")
            for fname, err in corrupt_files:
                log(f"    - {fname}: {err}")
        if empty_files:
            log(f"  [WARNING] {len(empty_files)} EMPTY FILES FOUND (Skipped days?):")
            for fname in empty_files:
                log(f"    - {fname}")
        if problematic_files:
            log(f"  [WARNING] {len(problematic_files)} FILES WITH LOGIC ISSUES FOUND:")
            for fname, issues in problematic_files:
                log(f"    - {fname} ({len(issues)} issues)")
            
    log(f"==================================================")

    if args.telegram:
        full_text = "\n".join(report_lines)
        log("\n[Telegram] Sending direct notification from host...")
        ok = send_telegram_alert(full_text)
        if ok:
            log("[Telegram] Alert sent successfully!")
        else:
            log("[Telegram] Alert failed to send.")

if __name__ == "__main__":
    main()
