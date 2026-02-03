import sys
import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, clear_output
from numba import cuda
from pathlib import Path

# Dynamic Root Finding
current_dir = Path(os.getcwd())
project_root = None

# Look for .git as a marker for the root
for parent in [current_dir] + list(current_dir.parents):
    if (parent / '.git').is_dir():
        project_root = parent
        break

if project_root:
    sys.path.append(str(project_root))
    print(f"Project Root Found: {project_root}")
else:
    # Fallback to current dir if not found (though less likely to work)
    project_root = current_dir
    sys.path.append(str(project_root))
    print("Project Root Not Found! Using current directory.")

print(f"Python Version: {sys.version.split()[0]}")

# Check CUDA
try:
    if cuda.is_available():
        print(f"🟢 CUDA Available: {cuda.detect()}")
    else:
        print("🟡 CUDA Not Available (using CPU)")
except Exception as e:
    print(f"🔴 CUDA Check Failed: {e}")

# Check Data
data_path = project_root / 'DATA' / 'RAW' / '*'
raw_files = glob.glob(str(data_path))
if raw_files:
    print(f"🟢 Data Files Found ({len(raw_files)}): {', '.join([os.path.basename(f) for f in raw_files])}")
else:
    print(f"🔴 No files in {data_path}")