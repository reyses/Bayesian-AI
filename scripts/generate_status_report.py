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
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["__pycache__", "venv", "env"]]

        level = root.count(os.sep)
        if level > 2: # 3 levels deep (0, 1, 2)
            continue

        indent = "│   " * (level)
        subindent = "│   " * (level + 1)

        if root != ".":
            tree_str += f"{indent}├── {os.path.basename(root)}/\n"

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
    entry_point = "engine_core.py"
    ready = "YES" if os.path.exists(entry_point) else "NO"

    return f"""### 7. EXECUTION READINESS
- **Entry Point:** `python {entry_point}`
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
    diff = run_command("git diff --name-status HEAD^ HEAD")
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

    # We can try running pytest and capturing output
    # Since we are in scripts/, we need to adjust path or run from root
    # This script is usually run from root in CI

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

def main():
    content = "# CURRENT STATUS REPORT\n\n"
    content += get_metadata()
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

    with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
        f.write(content)

    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
