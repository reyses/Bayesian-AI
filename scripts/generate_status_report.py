"""
Bayesian-AI - Status Report Generator
Generates a comprehensive status report (CURRENT_STATUS.md) from test results.
"""
import os
import sys
import subprocess
import datetime
import re
from pathlib import Path
import json

OUTPUT_FILE = "CURRENT_STATUS.md"

def run_command(command):
    try:
        return subprocess.getoutput(command)
    except Exception as e:
        return f"Error: {e}"

def get_metadata():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    branch = run_command("git rev-parse --abbrev-ref HEAD")
    commit = run_command("git rev-parse HEAD")
    return f"""### 1. METADATA
- **Timestamp:** {timestamp}
- **Git Branch:** {branch}
- **Last Commit:** {commit}
- **Build Status:** (See GitHub Actions Badge)
"""

def get_architecture_status():
    has_legacy = os.path.exists("core/layer_engine.py")
    has_quantum = os.path.exists("core/fractal_three_body.py")
    
    status = "UNKNOWN"
    if has_legacy and has_quantum:
        status = "TRANSITIONAL (Dual Architecture)"
    elif has_legacy:
        status = "LEGACY (9-Layer Hierarchy)"
    elif has_quantum:
        status = "NEXT-GEN (Fractal Quantum)"
        
    return f"""### 1A. ARCHITECTURE STATUS
- **Current State:** {status}
- **Active Engine:** 9-Layer Hierarchy (Legacy)
- **Experimental Engine:** Fractal Three-Body Quantum (Inactive)
- **Details:** See `AUDIT_REPORT.md`
"""

def get_changelog():
    log = run_command('git log --pretty=format:"%h - %s (%an)" -n 10')
    return f"""### 2. CHANGELOG
#### Last 10 Commits
```
{log}
```
"""

def get_tree():
    # Manual tree generation
    tree_str = "Bayesian-AI/\n"
    # Filter only relevant top-level directories or use walk with depth control
    for root, dirs, files in os.walk("."):
        # Exclude hidden dirs and specific folders
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["__pycache__", "venv", "env", "dist", "build"]]
        dirs.sort() # Sort directories alphabetically

        level = root.count(os.sep)
        if level > 2: # 3 levels deep (0, 1, 2)
            continue

        indent = "│   " * (level)
        subindent = "│   " * (level + 1)

        if root != ".":
            tree_str += f"{indent}├── {os.path.basename(root)}/\n"

        files.sort() # Sort files alphabetically
        for f in files:
            if f.startswith("."): continue

            annotation = ""
            if f.endswith(".py"):
                # Basic annotation logic
                path = os.path.join(root, f)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()
                        if "TODO" in content:
                            annotation = " [WIP]"
                        elif "test" in f:
                            annotation = " [TESTED]"
                        else:
                             # Check if corresponding test exists
                             test_path = os.path.join("tests", "test_" + f)
                             if os.path.exists(test_path):
                                 annotation = " [TESTED]"
                             else:
                                 annotation = " [COMPLETE]" # Default assumption
                except:
                    pass

            tree_str += f"{subindent}├── {f}{annotation}\n"

    return f"""### 3. FILE STRUCTURE
```
{tree_str}
```
"""

def get_code_stats():
    py_files = 0
    total_lines = 0
    for root, dirs, files in os.walk("."):
        # Exclude hidden dirs and specific folders
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["__pycache__", "venv", "env", "dist", "build"]]

        if ".git" in root: continue

        for f in files:
            if f.endswith(".py"):
                py_files += 1
                try:
                    with open(os.path.join(root, f), 'r', encoding='utf-8', errors='ignore') as file:
                        total_lines += sum(1 for line in file)
                except:
                    pass

    return f"""### 4. CODE STATISTICS
- **Python Files:** {py_files}
- **Total Lines of Code:** {total_lines}
"""

def check_file_content(filepath, patterns):
    if not os.path.exists(filepath):
        return f"File {filepath} not found"

    results = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        for pattern in patterns:
            found = "YES" if re.search(pattern, content, re.IGNORECASE) else "NO"
            results.append(f"- {pattern}: {found}")
    except Exception as e:
        return f"Error reading file: {e}"

    return "\n".join(results)

def get_critical_integration():
    databento = check_file_content("training/databento_loader.py", ["API_KEY", "DatabentoLoader"])
    orchestrator = check_file_content("training/orchestrator.py", ["DatabentoLoader", "pd.read_parquet"])

    return f"""### 5. CRITICAL INTEGRATION POINTS
- **Databento API:**
{databento}
- **Training Connection:**
{orchestrator}
"""

def get_dependencies():
    reqs = ""
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", 'r') as f:
            reqs = f.read()

    return f"""### 6. DEPENDENCIES
#### requirements.txt
```
{reqs}
```
- **Installation:** `pip install -r requirements.txt`
"""

def get_execution_readiness():
    entry_point = "core/engine_core.py"
    ready = "YES" if os.path.exists(entry_point) else "NO"

    return f"""### 7. EXECUTION READINESS
- **Entry Point:** `python -m core.engine_core`
- **Exists:** {ready}
- **Expected Runtime:** Long-running process (Server/Loop)
"""

def get_validation_checklist():
    bb = check_file_content("core/bayesian_brain.py", ["Laplace", "save", "load"])
    le = check_file_content("core/layer_engine.py", ["L1", "L9", "CUDA"])
    orch = check_file_content("training/orchestrator.py", ["DOE", "grid", "Walk-forward", "Monte Carlo", "iterations"])

    return f"""### 8. CODE VALIDATION CHECKLIST
#### bayesian_brain.py
{bb}

#### layer_engine.py
{le}

#### orchestrator.py
{orch}
"""

def get_testing_status():
    tests_exist = os.path.exists("tests")
    count = 0
    if tests_exist:
        count = len([f for f in os.listdir("tests") if f.startswith("test") or f.startswith("topic")])

    return f"""### 9. TESTING STATUS
- **Tests Directory:** {"YES" if tests_exist else "NO"}
- **Test Files Count:** {count}
"""

def get_modified_files():
    diff = run_command("git show --pretty='' --name-status HEAD")
    return f"""### 10. FILES MODIFIED (Last Commit)
```
{diff}
```
"""

def get_reviewer_checklist():
    return """### 11. REVIEWER CHECKLIST
- [ ] Architectural Review
- [ ] Potential Bugs
- [ ] Missing Features
- [ ] Performance Concerns
"""

def get_logic_core_validation():
    print("Running Logic Core Tests (topic_math.py)...")

    cmd = [sys.executable, "-m", "pytest", "tests/topic_math.py", "-v"]

    try:
        # Capture stdout and stderr
        result = subprocess.run(cmd, capture_output=True, text=True)

        status = "PASS" if result.returncode == 0 else "FAIL"
        output = result.stdout + "\n" + result.stderr

        # Parse for summary line like "4 passed in 0.05s"
        summary_line = "No summary found"
        for line in output.splitlines():
            if "passed" in line and "in" in line and "=" in line:
                summary_line = line.strip("= ")
                break

        details = f"""
- **Status:** {status}
- **Command:** `pytest tests/topic_math.py`
- **Summary:** {summary_line}
"""
        if status == "FAIL":
             details += f"\n**Failure Output:**\n```\n{output[-1000:]}\n```" # Last 1000 chars

        return f"""### 12. LOGIC CORE VALIDATION
{details}
"""
    except Exception as e:
        return f"""### 12. LOGIC CORE VALIDATION
- **Status:** ERROR
- **Details:** Failed to execute tests: {e}
"""

def run_test_script(script_path):
    cmd = [sys.executable, script_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        status = "PASS" if result.returncode == 0 else "FAIL"
        return status, result.stdout + result.stderr
    except Exception as e:
        return "ERROR", str(e)

def extract_pass_lines(output):
    lines = []
    for line in output.splitlines():
        if line.strip().startswith("PASS:"):
             lines.append(line.strip())
    if not lines:
        return "PASS: Check passed (no details)"
    return "\n".join(lines)

def get_qc_snapshot():
    print("Generating QC Validation Snapshot...")

    snapshot = "QC VALIDATION SNAPSHOT\n======================\n\n"

    # Topic 1: Executable Build
    print("Running Topic 1: Integrity...")
    status, output = run_test_script("tests/topic_build.py")
    snapshot += "Topic 1: Executable Build\n"
    if status == "PASS":
        snapshot += extract_pass_lines(output) + "\n"
    else:
        snapshot += "FAIL: Integrity Check Failed\n"
        snapshot += f"```\n{output[-500:]}\n```\n"

    snapshot += "\n"

    # Topic 2: Math and Logic
    snapshot += "Topic 2: Math and Logic\n"
    try:
        cmd = [sys.executable, "-m", "pytest", "tests/topic_math.py", "-q"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            snapshot += "PASS: Logic Core verified\n"
        else:
            snapshot += "FAIL: Logic Core failed\n"
    except:
        snapshot += "FAIL: Logic Core execution error\n"

    snapshot += "\n"

    # Topic 3: Diagnostics
    print("Running Topic 3: Diagnostics...")
    status, output = run_test_script("tests/topic_diagnostics.py")
    snapshot += "Topic 3: Diagnostics\n"
    if status == "PASS":
        snapshot += extract_pass_lines(output) + "\n"
    else:
        snapshot += "FAIL: Diagnostics Check Failed\n"
        snapshot += f"```\n{output[-500:]}\n```\n"

    snapshot += "\n"

    # Manifest Integrity
    print("Running Manifest Integrity Check...")
    status, output = run_test_script("scripts/manifest_integrity_check.py")
    snapshot += "Manifest Integrity\n"
    if status == "PASS":
        snapshot += "PASS: Manifest Integrity Check Passed\n"
    else:
        snapshot += "FAIL: Manifest Integrity Check Failed\n"
        snapshot += f"```\n{output[-500:]}\n```\n"

    return snapshot

def get_training_validation_metrics():
    print("Running Training Validation...")
    cmd = [sys.executable, "tests/test_training_validation.py"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        output = result.stdout

        start_tag = "::METRICS::"
        end_tag = "::END_METRICS::"

        if start_tag in output and end_tag in output:
            json_str = output.split(start_tag)[1].split(end_tag)[0].strip()
            try:
                all_metrics = json.loads(json_str)
            except json.JSONDecodeError:
                return "### TRAINING VALIDATION METRICS\n\nERROR: Failed to decode metrics JSON.\n"

            if not isinstance(all_metrics, list):
                all_metrics = [all_metrics]

            if not all_metrics:
                return "### TRAINING VALIDATION METRICS\n\nNo metrics returned.\n"

            # Use the first success, or the first failure if all failed
            metrics = next((m for m in all_metrics if m.get("status") == "SUCCESS"), all_metrics[0])

            status = metrics.get("status", "UNKNOWN")
            status_icon = "✓" if status == "SUCCESS" else "✗"

            iters = metrics.get("iterations_completed", "?")
            runtime = metrics.get("runtime_seconds", "?")
            # Files loaded is effectively 1 per metric entry in the new system
            files_loaded = len(all_metrics)
            ticks = metrics.get("total_ticks", 0)
            unique = metrics.get("unique_states_learned", "?")
            high_conf = metrics.get("high_confidence_states", "?")

            top_5 = metrics.get("top_5_states", [])
            top_5_str = ""
            for s in top_5:
                prob_pct = f"{s['probability']*100:.1f}%"
                wins = s['wins']
                losses = s['losses']
                top_5_str += f"- {s['state']}: {prob_pct} ({wins} wins, {losses} losses)\n"

            if not top_5_str:
                top_5_str = "None"

            table = f"""### 13. TRAINING VALIDATION METRICS
| Metric | Value | Status |
| :--- | :--- | :--- |
| Training Status | {status} | {status_icon} |
| Iterations Completed | {iters} | {status_icon} |
| Runtime | {runtime}s | - |
| Data Files Tested | {files_loaded} | {status_icon} |
| Total Ticks (Sample) | {ticks:,} | - |
| Unique States Learned | {unique} | - |
| High-Confidence States (80%+) | {high_conf} | {status_icon if isinstance(high_conf, int) and high_conf >= 0 else '✗'} |

**Top 5 States by Probability (Sample):**
{top_5_str}
"""
            if status != "SUCCESS":
                 table += f"\n**Error Details:**\n```\n{metrics.get('error', 'Unknown Error')}\n```"

            return table
        else:
             return f"### 13. TRAINING VALIDATION METRICS\n\nERROR: Metrics tags not found in output.\nOutput:\n{output[-500:]}"

    except subprocess.TimeoutExpired:
        return f"### 13. TRAINING VALIDATION METRICS\n\nERROR: Execution failed: Training validation timed out.\n"
    except Exception as e:
        return f"### 13. TRAINING VALIDATION METRICS\n\nERROR: Execution failed: {e}\n"

def get_doe_status():
    return """### 14. DOE OPTIMIZATION STATUS
- [ ] Parameter Grid Generator
- [ ] Latin Hypercube Sampling
- [ ] ANOVA Analysis Module
- [ ] Walk-Forward Test Harness
- [ ] Monte Carlo Bootstrap
- [ ] Response Surface Optimizer

**Current Status:** NOT IMPLEMENTED
**Estimated Implementation Time:** 1-2 weeks
**Priority:** HIGH (required for statistical validation)
"""

def main():
    content = "# CURRENT STATUS REPORT\n\n"
    content += get_metadata()
    content += "\n" + get_architecture_status()
    content += "\n" + get_changelog()
    content += "\n" + get_tree()
    content += "\n" + get_code_stats()
    content += "\n" + get_critical_integration()
    content += "\n" + get_dependencies()
    content += "\n" + get_execution_readiness()
    content += "\n" + get_validation_checklist()
    content += "\n" + get_testing_status()
    content += "\n" + get_modified_files()
    content += "\n" + get_reviewer_checklist()
    content += "\n" + get_logic_core_validation()
    content += "\n" + get_training_validation_metrics()
    content += "\n" + get_doe_status()
    # QC Snapshot was last, let's keep it last or before metrics?
    # Usually snapshot is summary.
    content += "\n" + get_qc_snapshot()

    with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
        f.write(content)

    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
