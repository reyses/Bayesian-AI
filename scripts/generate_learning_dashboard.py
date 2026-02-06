"""
Bayesian-AI - Learning Dashboard Generator
Generates a Jupyter notebook for the Full Learning Cycle with strict CUDA enforcement.
"""
import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🧠 Bayesian-AI Full Learning Cycle Dashboard\n",
    "\n",
    "**Objective:** Execute the full learning cycle on all available data with real-time visualization of P&L and Confidence.\n",
    "\n",
    "**Requirements:**\n",
    "*   **CUDA Required:** This notebook will fail if CUDA is not available.\n",
    "*   **Data:** Loads all `.dbn` and `.parquet` files from `DATA/RAW`.\n"
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
    "import ipywidgets as widgets\n",
    "from IPython.display import display, clear_output\n",
    "from numba import cuda\n",
    "\n",
    "# Add root to path\n",
    "current_dir = os.path.abspath('')\n",
    "project_root = os.path.dirname(current_dir) if 'notebooks' in current_dir else current_dir\n",
    "if project_root not in sys.path:\n",
    "    sys.path.append(project_root)\n",
    "\n",
    "from training.orchestrator import TrainingOrchestrator, load_data_from_directory, get_data_source\n",
    "from config.settings import RAW_DATA_PATH\n",
    "\n",
    "print(f\"Project Root: {project_root}\")\n",
    "print(f\"Data Path: {RAW_DATA_PATH}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Strict CUDA Enforcement 🛑\n",
    "The full learning cycle requires GPU acceleration. CPU fallback is strictly disabled."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    if not cuda.is_available():\n",
    "        raise RuntimeError(\"❌ CRITICAL: CUDA Required for Full Learning Cycle. CPU Fallback is disabled.\")\n",
    "    \n",
    "    device = cuda.get_current_device()\n",
    "    print(f\"🟢 CUDA Active: {device.name}\")\n",
    "    print(f\"   Compute Capability: {device.compute_capability}\")\n",
    "    \n",
    "except ImportError:\n",
    "    raise RuntimeError(\"❌ CRITICAL: Numba CUDA module not found. Please install CUDA drivers and Numba.\")\n",
    "except Exception as e:\n",
    "    raise RuntimeError(f\"❌ CRITICAL: CUDA Check Failed: {e}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load Training Data 📂\n",
    "Loading all historical data from `DATA/RAW`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    # Check if DATA/RAW exists\n",
    "    data_path = os.path.join(project_root, 'DATA', 'RAW')\n",
    "    if not os.path.exists(data_path):\n",
    "        # Fallback for some envs\n",
    "        data_path = 'DATA/RAW'\n",
    "        \n",
    "    print(f\"Loading data from: {data_path}...\")\n",
    "    full_data = load_data_from_directory(data_path)\n",
    "    print(f\"🟢 Data Loaded. Rows: {len(full_data):,}\")\n",
    "    \n",
    "    # Preview\n",
    "    display(full_data.head())\n",
    "    \n",
    "except Exception as e:\n",
    "    print(f\"🔴 Data Load Error: {e}\")\n",
    "    # Stop execution if no data\n",
    "    raise e"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Learning Dashboard 📊\n",
    "Real-time visualization of P&L and Confidence."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setup Widgets\n",
    "fig = go.FigureWidget()\n",
    "fig.add_scatter(name=\"Daily PnL\", y=[], mode='lines+markers', line=dict(color='green'))\n",
    "fig.add_scatter(name=\"Avg Confidence\", y=[], mode='lines', line=dict(color='blue', dash='dot'), yaxis='y2')\n",
    "\n",
    "fig.update_layout(\n",
    "    title=\"Learning Progress: PnL & Confidence\",\n",
    "    xaxis_title=\"Iteration\",\n",
    "    yaxis_title=\"PnL ($)\",\n",
    "    yaxis2=dict(\n",
    "        title=\"Avg Confidence\",\n",
    "        overlaying='y',\n",
    "        side='right',\n",
    "        range=[0, 1]\n",
    "    ),\n",
    "    height=500,\n",
    "    template=\"plotly_dark\"\n",
    ")\n",
    "\n",
    "progress_bar = widgets.IntProgress(\n",
    "    value=0,\n",
    "    min=0,\n",
    "    max=100,\n",
    "    description='Progress:',\n",
    "    bar_style='info',\n",
    "    style={'bar_color': '#00ff00'},\n",
    "    layout=widgets.Layout(width='100%')\n",
    ")\n",
    "\n",
    "status_label = widgets.Label(value=\"Ready to start.\")\n",
    "stats_output = widgets.Output()\n",
    "\n",
    "display(widgets.VBox([progress_bar, status_label, fig, stats_output]))\n",
    "\n",
    "def update_dashboard(metrics):\n",
    "    # Update Charts\n",
    "    with fig.batch_update():\n",
    "        fig.data[0].y = list(fig.data[0].y) + [metrics['pnl']]\n",
    "        fig.data[1].y = list(fig.data[1].y) + [metrics['average_confidence']]\n",
    "        fig.data[0].x = list(range(1, len(fig.data[0].y) + 1))\n",
    "        fig.data[1].x = list(range(1, len(fig.data[1].y) + 1))\n",
    "\n",
    "    # Update Progress\n",
    "    progress_bar.value = metrics['iteration']\n",
    "    progress_bar.max = metrics['total_iterations']\n",
    "    \n",
    "    status_label.value = f\"Iter {metrics['iteration']}/{metrics['total_iterations']} | PnL: ${metrics['pnl']:.2f} | WR: {metrics['win_rate']:.1%} | Conf: {metrics['average_confidence']:.2f}\"\n",
    "    \n",
    "    # Log to output (optional, keeping it clean)\n",
    "    # with stats_output:\n",
    "    #     print(f\"Iter {metrics['iteration']}: PnL=${metrics['pnl']:.2f}, Conf={metrics['average_confidence']:.3f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Execute Full Learning Cycle 🚀\n",
    "Run the training loop."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "ITERATIONS = 50 # Default for full cycle, user can adjust\n",
    "\n",
    "btn_start = widgets.Button(description=\"Start Training\", button_style='success', icon='play')\n",
    "\n",
    "def start_training(b):\n",
    "    btn_start.disabled = True\n",
    "    status_label.value = \"Initializing Orchestrator...\"\n",
    "    \n",
    "    try:\n",
    "        # Clear previous data\n",
    "        with fig.batch_update():\n",
    "            fig.data[0].y = []\n",
    "            fig.data[1].y = []\n",
    "        \n",
    "        # Initialize Orchestrator\n",
    "        # output_dir='models/production_learning' to separate from debug\n",
    "        orch = TrainingOrchestrator(\n",
    "            asset_ticker='MNQ', \n",
    "            data=full_data, \n",
    "            output_dir='models/production_learning',\n",
    "            use_gpu=True, # ENFORCE GPU\n",
    "            verbose=False\n",
    "        )\n",
    "        \n",
    "        status_label.value = \"Training Started...\"\n",
    "        \n",
    "        # Run\n",
    "        orch.run_training(iterations=ITERATIONS, on_progress=update_dashboard)\n",
    "        \n",
    "        status_label.value = \"✅ Training Complete! Model Saved.\"\n",
    "        progress_bar.bar_style = 'success'\n",
    "        \n",
    "    except Exception as e:\n",
    "        status_label.value = f\"❌ Error: {str(e)}\"\n",
    "        progress_bar.bar_style = 'danger'\n",
    "        with stats_output:\n",
    "            print(f\"Training Exception: {e}\")\n",
    "            import traceback\n",
    "            traceback.print_exc()\n",
    "    finally:\n",
    "        btn_start.disabled = False\n",
    "\n",
    "btn_start.on_click(start_training)\n",
    "display(btn_start)"
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

# Output to notebooks/learning_dashboard.ipynb
os.makedirs('notebooks', exist_ok=True)
with open('notebooks/learning_dashboard.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully: notebooks/learning_dashboard.ipynb")
