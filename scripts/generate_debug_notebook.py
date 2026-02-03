import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🐞 Debug Dashboard: Bayesian-AI System\n",
    "\n",
    "**Objective:** Rapid system validation and troubleshooting.\n",
    "**Status:** 🟢 Pass | 🔴 Fail | 🟡 Warning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Environment Check 🖥️\n",
    "Verify dependencies, CUDA availability, and data presence."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "import glob\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import plotly.graph_objects as go\n",
    "import plotly.express as px\n",
    "import ipywidgets as widgets\n",
    "from IPython.display import display, clear_output\n",
    "from numba import cuda\n",
    "\n",
    "# Add root to path\n",
    "sys.path.append(os.getcwd())\n",
    "\n",
    "print(f\"Python Version: {sys.version.split()[0]}\")\n",
    "\n",
    "# Check CUDA\n",
    "try:\n",
    "    if cuda.is_available():\n",
    "        print(f\"🟢 CUDA Available: {cuda.detect()}\")\n",
    "    else:\n",
    "        print(\"🟡 CUDA Not Available (using CPU)\")\n",
    "except Exception as e:\n",
    "    print(f\"🔴 CUDA Check Failed: {e}\")\n",
    "\n",
    "# Check Data\n",
    "raw_files = glob.glob('DATA/RAW/*')\n",
    "if raw_files:\n",
    "    print(f\"🟢 Data Files Found ({len(raw_files)}): {', '.join([os.path.basename(f) for f in raw_files])}\")\n",
    "else:\n",
    "    print(\"🔴 No files in DATA/RAW/\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Pipeline Test 📊\n",
    "Load a sample file and visualize to ensure data integrity."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from training.orchestrator import get_data_source\n",
    "\n",
    "try:\n",
    "    if not raw_files:\n",
    "        raise FileNotFoundError(\"No data files to test\")\n",
    "        \n",
    "    target_file = raw_files[0]\n",
    "    print(f\"Loading {target_file}...\")\n",
    "    \n",
    "    df = get_data_source(target_file)\n",
    "    \n",
    "    print(f\"🟢 Load Success! Rows: {len(df)}\")\n",
    "    print(f\"Columns: {list(df.columns)}\")\n",
    "    display(df.head())\n",
    "    \n",
    "    # Simple Plot\n",
    "    plot_df = df.head(1000)\n",
    "    fig = go.Figure(data=[go.Candlestick(x=plot_df.index if 'timestamp' not in plot_df.columns else plot_df['timestamp'],\n",
    "                open=plot_df['open'] if 'open' in plot_df.columns else plot_df['price'],\n",
    "                high=plot_df['high'] if 'high' in plot_df.columns else plot_df['price'],\n",
    "                low=plot_df['low'] if 'low' in plot_df.columns else plot_df['price'],\n",
    "                close=plot_df['close'] if 'close' in plot_df.columns else plot_df['price'])])\n",
    "    fig.update_layout(title=f'Price Sample: {os.path.basename(target_file)}', xaxis_rangeslider_visible=False)\n",
    "    fig.show()\n",
    "    \n",
    "except Exception as e:\n",
    "    print(f\"🔴 Data Pipeline Failed: {e}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Core Component Tests ⚙️\n",
    "Isolated tests for key modules."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from core.state_vector import StateVector\n",
    "from core.bayesian_brain import BayesianBrain\n",
    "from core.layer_engine import LayerEngine\n",
    "from cuda_modules.velocity_gate import CUDAVelocityGate\n",
    "import time\n",
    "\n",
    "print(\"--- Component Status ---\")\n",
    "\n",
    "# 1. StateVector\n",
    "try:\n",
    "    sv = StateVector.null_state()\n",
    "    assert hash(sv) is not None\n",
    "    print(\"🟢 StateVector: Operational\")\n",
    "except Exception as e:\n",
    "    print(f\"🔴 StateVector: Failed ({e})\")\n",
    "\n",
    "# 2. BayesianBrain\n",
    "try:\n",
    "    bb = BayesianBrain()\n",
    "    if os.path.exists('probability_table.pkl'):\n",
    "        bb.load('probability_table.pkl')\n",
    "        print(f\"🟢 BayesianBrain: Loaded {len(bb.table)} states from disk\")\n",
    "    else:\n",
    "        print(\"🟡 BayesianBrain: No existing table found (Clean Start)\")\n",
    "except Exception as e:\n",
    "    print(f\"🔴 BayesianBrain: Failed ({e})\")\n",
    "\n",
    "# 3. LayerEngine\n",
    "try:\n",
    "    le = LayerEngine(use_gpu=False) # Test CPU init first\n",
    "    print(\"🟢 LayerEngine: Initialized\")\n",
    "except Exception as e:\n",
    "    print(f\"🔴 LayerEngine: Failed ({e})\")\n",
    "\n",
    "# 4. CUDA VelocityGate\n",
    "try:\n",
    "    vg = CUDAVelocityGate(use_gpu=True)\n",
    "    print(f\"🟢 VelocityGate: Initialized (GPU={vg.use_gpu})\")\n",
    "except Exception as e:\n",
    "    print(f\"🔴 VelocityGate: Failed ({e})\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Mini Training Run 🏃\n",
    "Execute `orchestrator.py` in a subprocess with live output."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import subprocess\n",
    "import threading\n",
    "\n",
    "def run_training_process(iterations=5):\n",
    "    os.makedirs('debug_outputs', exist_ok=True)\n",
    "    cmd = [\n",
    "        \"python\", \"training/orchestrator.py\",\n",
    "        \"--data-dir\", \"DATA/RAW\",\n",
    "        \"--iterations\", str(iterations),\n",
    "        \"--output\", \"debug_outputs/\"\n",
    "    ]\n",
    "    \n",
    "    print(f\"Executing: {' '.join(cmd)}\")\n",
    "    \n",
    "    process = subprocess.Popen(\n",
    "        cmd,\n",
    "        stdout=subprocess.PIPE,\n",
    "        stderr=subprocess.STDOUT,\n",
    "        text=True,\n",
    "        bufsize=1\n",
    "    )\n",
    "    \n",
    "    return process\n",
    "\n",
    "btn_run = widgets.Button(description=\"Start Mini Run (5 Iters)\", button_style='success')\n",
    "out_log = widgets.Output(layout={'border': '1px solid black', 'height': '300px', 'overflow_y': 'scroll'})\n",
    "\n",
    "def on_run_click(b):\n",
    "    out_log.clear_output()\n",
    "    with out_log:\n",
    "        print(\"Starting subprocess...\")\n",
    "        proc = run_training_process()\n",
    "        \n",
    "        while True:\n",
    "            line = proc.stdout.readline()\n",
    "            if not line and proc.poll() is not None:\n",
    "                break\n",
    "            if line:\n",
    "                print(line.strip())\n",
    "        \n",
    "        if proc.returncode == 0:\n",
    "            print(\"\\n✅ Process Completed Successfully\")\n",
    "        else:\n",
    "            print(f\"\\n❌ Process Failed with Code {proc.returncode}\")\n",
    "\n",
    "btn_run.on_click(on_run_click)\n",
    "display(btn_run, out_log)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Probability Table Analysis 📈\n",
    "Analyze the learned states from the run."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "\n",
    "def analyze_table(filepath):\n",
    "    if not os.path.exists(filepath):\n",
    "        print(f\"🔴 File not found: {filepath}\")\n",
    "        return\n",
    "        \n",
    "    with open(filepath, 'rb') as f:\n",
    "        data = pickle.load(f)\n",
    "    \n",
    "    table = data['table']\n",
    "    print(f\"Loaded {len(table)} states.\")\n",
    "    \n",
    "    # Convert to DataFrame for analysis\n",
    "    records = []\n",
    "    for state, stats in table.items():\n",
    "        total = stats['total']\n",
    "        wins = stats['wins']\n",
    "        if total > 0:\n",
    "            records.append({\n",
    "                'total': total,\n",
    "                'wins': wins,\n",
    "                'win_rate': wins/total,\n",
    "                'L1': state.L1_bias,\n",
    "                'L5': state.L5_trend\n",
    "            })\n",
    "            \n",
    "    if not records:\n",
    "        print(\"No records to analyze.\")\n",
    "        return\n",
    "        \n",
    "    df_stats = pd.DataFrame(records)\n",
    "    \n",
    "    # Histogram of sample sizes\n",
    "    fig1 = px.histogram(df_stats, x='total', nbins=50, title='Distribution of Samples per State')\n",
    "    fig1.show()\n",
    "    \n",
    "    # Win Rate vs Sample Size\n",
    "    fig2 = px.scatter(df_stats, x='total', y='win_rate', title='Win Rate vs Sample Size',\n",
    "                      hover_data=['L1', 'L5'])\n",
    "    fig2.show()\n",
    "    \n",
    "    # High Confidence States\n",
    "    high_conf = df_stats[df_stats['total'] > 10].sort_values('win_rate', ascending=False).head(10)\n",
    "    print(\"Top 10 High Confidence States:\")\n",
    "    display(high_conf)\n",
    "\n",
    "# Check debug output first, then main\n",
    "if os.path.exists('debug_outputs/probability_table.pkl'):\n",
    "    analyze_table('debug_outputs/probability_table.pkl')\n",
    "elif os.path.exists('probability_table.pkl'):\n",
    "    analyze_table('probability_table.pkl')\n",
    "else:\n",
    "    print(\"🟡 No probability table found. Run training first.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Performance Profiling ⏱️\n",
    "Benchmark key operations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import timeit\n",
    "\n",
    "print(\"Benchmarking StateVector Hashing...\")\n",
    "setup_code = \"from core.state_vector import StateVector; sv = StateVector.null_state()\"\n",
    "t = timeit.timeit(\"hash(sv)\", setup=setup_code, number=100000)\n",
    "print(f\"Create & Hash 100k states: {t:.4f}s\")\n",
    "\n",
    "print(\"\\nBenchmarking VelocityGate (CPU fallback check)...\")\n",
    "setup_vg = \"\"\"\n",
    "from cuda_modules.velocity_gate import CUDAVelocityGate\n",
    "import numpy as np\n",
    "vg = CUDAVelocityGate(use_gpu=False)\n",
    "prices = np.random.random(100).astype(np.float32)\n",
    "\"\"\"\n",
    "t_vg = timeit.timeit(\"vg.detect_cascade(prices)\", setup=setup_vg, number=1000)\n",
    "print(f\"1000 detections (CPU): {t_vg:.4f}s\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. DOE Simulation Preview 🧪\n",
    "Preview parameter combinations for optimization."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import itertools\n",
    "\n",
    "param_grid = {\n",
    "    'min_prob': [0.70, 0.75, 0.80, 0.85],\n",
    "    'min_conf': [0.20, 0.30, 0.40],\n",
    "    'stop_loss': [10, 20, 30],\n",
    "    'kill_zones': ['tight', 'wide']\n",
    "}\n",
    "\n",
    "keys, values = zip(*param_grid.items())\n",
    "combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]\n",
    "\n",
    "print(f\"Total Combinations: {len(combinations)}\")\n",
    "print(\"Sample First 5:\")\n",
    "for c in combinations[:5]:\n",
    "    print(c)\n",
    "\n",
    "print(\"\\nNOTE: Full DOE integration requires implementing the grid search loop in orchestrator.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Quick Fixes & Utilities 🛠️"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import shutil\n",
    "\n",
    "def clean_pycache():\n",
    "    print(\"Cleaning __pycache__...\")\n",
    "    count = 0\n",
    "    for root, dirs, files in os.walk('.'):\n",
    "        for d in dirs:\n",
    "            if d == '__pycache__':\n",
    "                shutil.rmtree(os.path.join(root, d))\n",
    "                count += 1\n",
    "    print(f\"Removed {count} directories.\")\n",
    "\n",
    "btn_clean = widgets.Button(description=\"Clear PyCache\")\n",
    "btn_clean.on_click(lambda b: clean_pycache())\n",
    "display(btn_clean)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('debug_dashboard.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully: debug_dashboard.ipynb")
